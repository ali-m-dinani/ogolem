"""Evaluation-only engines and a single-allocation Slurm launcher for the bridge."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

try:
    import bridge
except ModuleNotFoundError:
    import reaxff_bridge as bridge


SLURM = '''#!/bin/bash
#SBATCH --job-name=ff_opt
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=open
#SBATCH --partition=basic
#SBATCH --constraint=genoa

set -euo pipefail
module load ams/2023.103
module load intel/2023.0
module load mpi/2021.8.0
unset SCM_DISABLE_MPI
export NSCM="$SLURM_NTASKS"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE
export SCM_USE_THREADS="$OMP_NUM_THREADS"
export OMP_STACKSIZE=1G
printf 'Job %s | Nodes %s | MPI ranks %s | Threads/rank %s\\n' "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$NSCM" "$OMP_NUM_THREADS"
scontrol show job "$SLURM_JOB_ID"
lscpu
cd "$SLURM_SUBMIT_DIR"
: "${OGOLEM_JAR:?Set OGOLEM_JAR to the Linux-compatible OGolem jar}"
export REAXFF_BRIDGE_PYTHON="${REAXFF_BRIDGE_PYTHON:-python3}"
export OGO_EXTCALLCMD="$PWD/ogo_evaluator.sh"
chmod +x "$OGO_EXTCALLCMD"
"$REAXFF_BRIDGE_PYTHON" bridge.py check --bundle .
"${JAVA_BIN:-java}" -Xmx4g --add-modules=jdk.incubator.vector --enable-preview -jar "$OGOLEM_JAR" --adaptive fit.ogo 1
'''


def configure_backend(bundle, backend, engine_python='{python}', ams_job_file=None):
    bundle = Path(bundle)
    shutil.copyfile(__file__, bundle / 'engines.py')
    if backend == 'custom':
        return
    config = {'timeout_seconds': 3500}
    if backend == 'ams':
        config.update(executable='${AMSBIN}/reaxff', job_file=None)
        if ams_job_file is not None:
            (bundle / 'engine-job.sh').write_bytes(bridge.read_text(ams_job_file).replace('\r\n', '\n').encode('utf-8'))
            config['job_file'] = 'engine-job.sh'
        (bundle / 'submit.slurm').write_bytes(SLURM.encode('utf-8'))
        (bundle / 'submit-evaluate.slurm').write_bytes(
            SLURM.replace('--adaptive fit.ogo 1\n', '--adaptive fit.ogo 1 params=initial.params\n').encode('utf-8'))
    else:
        config.update(energy_minimization_steps=None, energy_minimization_learning_rate=0.001,
                      end_rmsg=None, cutoff2=None, bonded_cutoff=5.0, max_num_clusters=10,
                      num_threads=1,
                      note='Requires the JAX-ReaxFF-compatible jax_md.reaxff API; control is not read by JAX-ReaxFF. See tools/REAXFF_BRIDGE.md.')
    bridge.write_json(bundle / 'engine.json', config)
    bridge.write_json(bundle / 'evaluator.json', {
        'command': [engine_python, '{bundle}/engines.py', backend],
        'timeout_seconds': 3600,
        'note': 'Evaluation only. AMS runs within an existing allocation; submit.slurm launches OGolem once, not sbatch per candidate.'})


def control_values(text):
    values = {}
    for line in text.splitlines():
        fields = bridge.data_part(line).split()
        if len(fields) >= 2:
            values[fields[1].lower()] = bridge.number(fields[0])
    return values


def evaluation_control(text):
    lines = []
    for line in text.splitlines():
        fields = bridge.data_part(line).split()
        if len(fields) < 2 or fields[1].lower() not in ('ffoact', 'igeopt'):
            lines.append(line)
    return '\n'.join(lines) + '\n3 ffoact\n0 igeopt\n'


def parse_fort99(text, targets):
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            row = [bridge.number(value) for value in fields[-5:]]
        except ValueError:
            continue
        rows.append(row)
    if len(rows) != len(targets):
        raise bridge.InputError(f'fort.99 has {len(rows)} numeric target rows; expected {len(targets)}. Refusing incomplete/unsupported report.')
    expected = Counter((round(target['target'], 4), round(target['weight'], 4)) for target in targets)
    observed = Counter((round(row[1], 4), round(row[2], 4)) for row in rows)
    if observed != expected:
        raise bridge.InputError('fort.99 targets/weights do not match the complete training set')
    previous = 0.0
    for prediction, target, weight, error, cumulative in rows:
        if weight <= 0 or error < 0 or cumulative < 0:
            raise bridge.InputError('fort.99 contains invalid error values')
        if not math.isclose(cumulative - previous, error, rel_tol=1e-5, abs_tol=0.001):
            raise bridge.InputError('fort.99 cumulative error is inconsistent')
        previous = cumulative
    return {'fitness': previous, 'engine': 'ams-standalone', 'target_count': len(rows),
            'precision_note': 'Fitness is the cumulative error printed by standalone ReaxFF in fort.99.'}


def ams_evaluate(bundle, manifest, config, workdir):
    for name in ('fort.99', 'fort.13'):
        if (workdir / name).exists():
            raise bridge.InputError(f'Refusing stale AMS output: {name}')
    original_control = bridge.read_text(workdir / 'control') if (workdir / 'control').exists() else ''
    (workdir / 'control').write_text(evaluation_control(original_control), encoding='utf-8')
    (workdir / 'iopt').write_text('6\n', encoding='utf-8')
    (workdir / 'params').write_text(''.join(
        f'{item["section"]} {item["entry"]} {item["parameter"]}\n' for item in manifest['parameters']), encoding='utf-8')
    if config.get('job_file'):
        job_file = (bundle / config['job_file']).resolve()
        if job_file.parent != bundle:
            raise bridge.InputError('AMS job_file must reside directly inside the bundle')
        command = ['bash', str(job_file)]
    else:
        executable = os.path.expandvars(config['executable'])
        if '$' in executable:
            raise bridge.InputError('AMSBIN is not set; load the AMS environment first')
        command = [executable]
    timeout = bridge.number(str(config.get('timeout_seconds', 3500)))
    if timeout <= 0:
        raise bridge.InputError('Engine timeout must be positive')
    with (workdir / 'engine.stdout.log').open('wb') as stdout, (workdir / 'engine.stderr.log').open('wb') as stderr:
        subprocess.run(command, cwd=workdir, stdout=stdout, stderr=stderr, check=True, timeout=timeout)
    return parse_fort99(bridge.read_text(workdir / 'fort.99'), manifest['targets'])


def jax_settings(manifest, config, workdir):
    unsupported = {item['section'] for item in manifest['targets']} - {'ENERGY', 'CHARGE', 'GEOMETRY'}
    if unsupported:
        raise bridge.InputError(f'JAX backend does not support these targets: {sorted(unsupported)}')
    for item in manifest['targets']:
        if item['section'] == 'GEOMETRY' and len(item['atoms']) not in (2, 3, 4):
            raise bridge.InputError('JAX backend requires explicit distance/angle/torsion targets')
    control = control_values(bridge.read_text(workdir / 'control')) if (workdir / 'control').exists() else {}
    for key, expected in [('icelop', 0), ('icharg', 4), ('ichaen', 1)]:
        if control.get(key, expected) != expected:
            raise bridge.InputError(f'JAX backend cannot reproduce {key}={control[key]}; requires {expected}')
    if control.get('imetho', 1) not in (0, 1):
        raise bridge.InputError('JAX backend supports single point or geometry minimization, not MD')
    settings = dict(config)
    defaults = {'energy_minimization_steps': int(control.get('imaxit', 0)) if control.get('imetho', 1) == 1 else 0,
                'end_rmsg': control.get('endmm', 1.0), 'cutoff2': control.get('cutof2', 0.001)}
    for key, value in defaults.items():
        if settings.get(key) is None:
            settings[key] = value
    if settings['energy_minimization_steps'] < 0 or int(settings['energy_minimization_steps']) != settings['energy_minimization_steps']:
        raise bridge.InputError('energy_minimization_steps must be a nonnegative integer')
    for key in ('end_rmsg', 'cutoff2', 'energy_minimization_learning_rate', 'bonded_cutoff'):
        if bridge.number(str(settings[key])) <= 0:
            raise bridge.InputError(f'{key} must be positive')
    if not isinstance(settings['num_threads'], int) or settings['num_threads'] < 1:
        raise bridge.InputError('num_threads must be a positive integer')
    if not isinstance(settings['max_num_clusters'], int) or not 1 <= settings['max_num_clusters'] <= 15:
        raise bridge.InputError('max_num_clusters must be an integer from 1 to 15')
    for geometry in manifest['geometries'].values():
        if [atom['id'] for atom in geometry['atoms']] != list(range(1, len(geometry['atoms']) + 1)):
            raise bridge.InputError('JAX backend requires consecutive atom IDs starting at one')
    raw_geo = bridge.read_text(workdir / 'geo')
    if re.search(r'^\s*(?:ATOM\s|FIX|VELOCITY|TAPER|CELL RESTRAINT)', raw_geo, re.M | re.I):
        raise bridge.InputError('JAX backend encountered unsupported geometry directives')
    return settings


def canonical_training(targets):
    lines = []
    for section in ('ENERGY', 'CHARGE', 'GEOMETRY'):
        items = [item for item in targets if item['section'] == section]
        if not items:
            continue
        lines.append(section)
        for item in items:
            if section == 'ENERGY':
                expression = ' '.join(f'{"+" if term["coefficient"] > 0 else "-"} {term["name"]} / {1 / abs(term["coefficient"]):.17g}' for term in item['terms'])
                lines.append(f'{item["weight"]:.17g} {expression} {item["target"]:.17g}')
            else:
                atoms = ' '.join(map(str, item['atoms']))
                lines.append(f'{item["name"]} {item["weight"]:.17g} {atoms} {item["target"]:.17g}')
        lines.append('END' + section)
    return '\n'.join(lines) + '\n'


def jax_evaluate(bundle, manifest, config, workdir):
    settings = jax_settings(manifest, config, workdir)
    import numpy as numpy
    from jaxreaxff import driver

    records = []
    original_report = driver.produce_error_report
    original_cluster = driver.process_and_cluster_geos
    original_argv = sys.argv

    def capture_report(filename, training, errors, names):
        count = sum(numpy.asarray(values[2]).size for values in errors.values())
        if count != len(manifest['targets']):
            raise bridge.InputError(f'JAX evaluated {count} targets, expected {len(manifest["targets"])}')
        loss = math.fsum(float(value) for values in errors.values() for value in numpy.asarray(values[2]).flat)
        records.append({'fitness': loss, 'engine': 'jax-md-via-jax-reaxff', 'target_count': count,
                        'settings': settings})
        original_report(filename, training, errors, names)

    def cluster_with_limit(*args, **kwargs):
        kwargs['num_threads'] = settings['num_threads']
        kwargs['max_num_clusters'] = min(kwargs['max_num_clusters'], len(args[0]))
        return original_cluster(*args, **kwargs)

    (workdir / 'jax-trainset.in').write_text(canonical_training(manifest['targets']), encoding='utf-8')
    (workdir / 'params').write_text(''.join(
        f'{item["section"]} {item["entry"]} {item["parameter"]} {item["interval"]:.17g} {item["lower"]:.17g} {item["upper"]:.17g}\n'
        for item in manifest['parameters']), encoding='utf-8')
    sys.argv = ['jax-evaluator', '--init_FF', 'ffield', '--params', 'params', '--geo', 'geo',
                '--train_file', 'jax-trainset.in', '--num_trials', '0', '--num_steps', '0',
                '--init_FF_type', 'fixed', '--out_folder', 'jax-output', '--save_opt', 'all']
    for option, key in [('--num_e_minim_steps', 'energy_minimization_steps'),
                        ('--e_minim_LR', 'energy_minimization_learning_rate'), ('--end_RMSG', 'end_rmsg'),
                        ('--cutoff2', 'cutoff2'), ('--bonded_cutoff', 'bonded_cutoff'),
                        ('--max_num_clusters', 'max_num_clusters')]:
        sys.argv.extend([option, str(settings[key])])
    try:
        driver.produce_error_report = capture_report
        driver.process_and_cluster_geos = cluster_with_limit
        driver.main()
    finally:
        driver.produce_error_report = original_report
        driver.process_and_cluster_geos = original_cluster
        sys.argv = original_argv
    if len(records) != 1:
        raise bridge.InputError('JAX driver must return exactly one evaluation report')
    return records[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=('ams', 'jax-md'))
    args = parser.parse_args()
    try:
        workdir = Path.cwd()
        if (workdir / 'fitness.json').exists():
            raise bridge.InputError('Refusing existing fitness.json')
        request = json.loads(bridge.read_text(workdir / 'request.json'))
        bundle, manifest, parameters = bridge.load_bundle(Path(request['manifest']).parent)
        config = json.loads(bridge.read_text(bundle / 'engine.json'))
        evaluator = ams_evaluate if args.backend == 'ams' else jax_evaluate
        response = evaluator(bundle, manifest, config, workdir)
        bridge.response_fitness(response, manifest['targets'])
        bridge.write_json(workdir / 'fitness.json', response)
    except (ValueError, OSError, KeyError, ImportError, subprocess.SubprocessError) as error:
        parser.exit(2, f'ENGINE ERROR: {error}\n')


if __name__ == '__main__':
    main()
