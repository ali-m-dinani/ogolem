import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_reaxff_bridge as fixtures
import reaxff_bridge as bridge
import reaxff_engines as engines


class EngineTest(unittest.TestCase):
    setUp = fixtures.BridgeTest.setUp
    bundle = fixtures.BridgeTest.bundle
    candidate = fixtures.BridgeTest.candidate
    configure_mock = fixtures.BridgeTest.configure_mock

    def test_mask_bounds_and_ams_general_pointer(self):
        self.assertEqual(bridge.read_parameters('1 2 0 0.1 0 3')[0][0].key, (1, 2, 1))
        (self.source / 'params').unlink()
        field = bridge.ForceField(fixtures.force_field_text())
        parameters = [bridge.Parameter(*key, 0, 0, 5, 1) for key in field.locations]
        for name, values in [('ffield_bool', [1.0 if index < 3 else 0.0 for index in range(len(parameters))]),
                             ('ffield_min', [0.0] * len(parameters)), ('ffield_max', [5.0] * len(parameters))]:
            (self.source / name).write_text(field.render(parameters, values))
        bundle = self.bundle()
        _, manifest, selected = bridge.load_bundle(bundle)
        self.assertEqual(len(selected), 3)
        self.assertIn('params', manifest['sha256'])
        self.assertIn('ffield_bool', manifest['sha256'])
        (self.source / 'ffield_bool').write_text(field.render(parameters, [0.5] * len(parameters)))
        with self.assertRaisesRegex(bridge.InputError, 'zero or one'):
            bridge.source_parameters(self.source)

    def test_ams_preparation_with_mock_engine(self):
        bundle = self.root / 'ams-bundle'
        bridge.convert(self.source, bundle, backend='ams')
        mock = self.root / 'fake_ams.py'
        mock.write_text('from pathlib import Path\n'
                        'assert Path("iopt").read_text().strip() == "6"\n'
                        'assert all(len(line.split()) == 3 for line in Path("params").read_text().splitlines())\n'
                        'assert "3 ffoact" in Path("control").read_text()\n'
                        'Path("fort.99").write_text("FField value QM/Lit value Weight Error Total error\\nEnergy +molecule/2 -molecule/1 0.0 -1.0 0.5 4.0 4.0\\n")\n')
        config = json.loads((bundle / 'engine.json').read_text())
        config['executable'] = sys.executable
        bridge.write_json(bundle / 'engine.json', config)
        workdir = self.candidate(bundle)
        _, manifest, _ = bridge.load_bundle(bundle)
        for name in ('control', 'params', 'geo', 'trainset.in', 'ffield'):
            shutil.copyfile(bundle / 'inputs' / name, workdir / name)
        real_run = subprocess.run

        def mock_command(command, **kwargs):
            return real_run([sys.executable, str(mock)], **kwargs)

        with patch.object(engines.subprocess, 'run', side_effect=mock_command):
            response = engines.ams_evaluate(bundle, manifest, config, workdir)
        self.assertEqual(response['fitness'], 4)
        with self.assertRaisesRegex(bridge.InputError, 'stale'):
            engines.ams_evaluate(bundle, manifest, config, workdir)
        self.assertEqual((bundle / 'inputs/iopt').read_text().strip(), '4')
        self.assertNotIn(b'\r', (bundle / 'submit.slurm').read_bytes())
        self.assertNotIn(b'\r', (bundle / 'ogo_evaluator.sh').read_bytes())
        self.assertIn('--adaptive fit.ogo 1', (bundle / 'submit.slurm').read_text())
        self.assertIn('--nodes=2', (bundle / 'submit.slurm').read_text())
        self.assertIn('params=initial.params', (bundle / 'submit-evaluate.slurm').read_text())

    def test_masks_map_shifted_interaction_rows_and_reject_ambiguity(self):
        (self.source / 'params').unlink()
        field = bridge.ForceField(fixtures.force_field_text())
        parameters = [bridge.Parameter(*key, 0, 0, 5, 1) for key in field.locations]
        flags = [1.0 if item.key == (6, 1, 1) else 0.0 for item in parameters]
        for name, values in [('ffield_bool', flags), ('ffield_min', [0.0] * len(parameters)),
                             ('ffield_max', [5.0] * len(parameters))]:
            (self.source / name).write_text(field.render(parameters, values))
        extra = ' 0 1 1 0' + ''.join(f'{2.0:10.4f}' for index in range(7)) + '\n'
        shifted = fixtures.force_field_text().replace('1 ! torsions\n', '2 ! torsions\n' + extra)
        (self.source / 'ffield').write_text(shifted)
        selected, warnings = bridge.source_parameters(self.source)
        self.assertEqual(selected[0].key, (6, 2, 1))
        self.assertEqual(len(warnings), 1)
        (self.source / 'ffield').write_text(shifted.replace(' 0 1 1 0', ' 1 1 1 1'))
        with self.assertRaisesRegex(bridge.InputError, 'ambiguous'):
            bridge.source_parameters(self.source)

    def test_ams_report_rejects_incomplete_wrong_and_nonfinite_data(self):
        targets = [{'target': -1, 'weight': 0.5}, {'target': 2, 'weight': 1}]
        valid = 'Energy a 0 -1 .5 4 4\nEnergy b 3 2 1 1 5\n'
        self.assertEqual(engines.parse_fort99(valid, targets)['fitness'], 5)
        for text in [valid.splitlines()[0], valid.replace(' 2 1 ', ' 9 1 '),
                     valid.replace('1 5', '1 50'), valid.replace('3 2', 'NaN 2'), valid + valid]:
            with self.subTest(text=text), self.assertRaises(bridge.InputError):
                engines.parse_fort99(text, targets)

    def test_control_replaces_optimization_modes_not_relaxation_settings(self):
        original = '4 ffoact\n1 igeopt\n100 imaxit\n2 endmm\n1 imetho\n'
        result = engines.control_values(engines.evaluation_control(original))
        self.assertEqual(result, {'ffoact': 3, 'igeopt': 0, 'imaxit': 100, 'endmm': 2, 'imetho': 1})

    def test_custom_job_is_copied_not_submitted(self):
        job = self.root / 'job.sh'
        job.write_bytes(b'#!/bin/bash\r\n"$AMSBIN/reaxff"\r\n')
        bundle = self.root / 'custom-job'
        bridge.convert(self.source, bundle, backend='ams', ams_job_file=job)
        self.assertNotIn(b'\r', (bundle / 'engine-job.sh').read_bytes())
        self.assertEqual(json.loads((bundle / 'engine.json').read_text())['job_file'], 'engine-job.sh')

    def test_jax_settings_and_canonical_training(self):
        bundle = self.root / 'jax-bundle'
        bridge.convert(self.source, bundle, backend='jax-md')
        _, manifest, _ = bridge.load_bundle(bundle)
        config = json.loads((bundle / 'engine.json').read_text())
        (self.source / 'control').write_text('1 imetho\n123 imaxit\n2 endmm\n0.0001 cutof2\n')
        settings = engines.jax_settings(manifest, config, self.source)
        self.assertEqual(settings['energy_minimization_steps'], 123)
        self.assertEqual(settings['end_rmsg'], 2)
        canonical = engines.canonical_training(manifest['targets'])
        targets = bridge.read_training(canonical, manifest['geometries'])
        self.assertEqual(targets[0]['terms'], manifest['targets'][0]['terms'])
        (self.source / 'control').write_text('1 icelop\n')
        with self.assertRaisesRegex(bridge.InputError, 'icelop'):
            engines.jax_settings(manifest, config, self.source)
        manifest['targets'][0]['section'] = 'CELL'
        with self.assertRaisesRegex(bridge.InputError, 'does not support'):
            engines.jax_settings(manifest, config, self.source)

    def test_portable_evaluator_placeholders(self):
        bundle = self.root / 'ams-portable'
        bridge.convert(self.source, bundle, backend='ams')
        command, timeout = bridge.evaluator_config(bundle)
        self.assertEqual(command, [sys.executable, str(bundle) + '/engines.py', 'ams'])
        self.assertGreater(timeout, 0)

    def test_s4_profile_runs_a_small_actual_ogolem_ga(self):
        root = Path(__file__).resolve().parents[2]
        if os.name != 'nt' or not (root / 'build/libs/ogolem-snapshot.jar').exists():
            self.skipTest('Built Windows OGolem required')
        (self.source / 'params').write_text('1 1 1 0.1 0 5\n1 2 1 0.1 0 5\n2 1 1 0.1 0 5\n')
        bundle = self.root / 's4'
        bridge.convert(self.source, bundle, profile='s4-inspired', iterations=2, population=4)
        self.configure_mock(bundle)
        environment = dict(os.environ, OGO_EXTCALLCMD=str(bundle / 'ogo_evaluator.cmd'), REAXFF_BRIDGE_PYTHON=sys.executable)
        result = subprocess.run([str(root / 'ogolem.cmd'), '--adaptive', 'fit.ogo', '1'],
                                cwd=bundle, env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('Exception', result.stderr)
        self.assertGreaterEqual(len(list((bundle / 'evaluations').iterdir())), 4)
        self.assertTrue(any(bundle.glob('fit/*')))

    @unittest.skipUnless(os.environ.get('JAX_REAXFF_TEST_PYTHON'), 'Optional real JAX evaluation')
    def test_real_jax_evaluation_without_parameter_optimization(self):
        root = Path(__file__).resolve().parents[2].parent / 'JAX-ReaxFF'
        shutil.copyfile(root / 'Datasets/cobalt/ffield_lit', self.source / 'ffield')
        (self.source / 'params').write_text('3 1 1 0.1 -500 500\n')
        (self.source / 'geo').write_text(
            'BIOGRF 200\nDESCRP cobalt\nRUTYPE SINGLE POINT\n'
            'HETATM 1 Co 0.0 0.0 0.0 Co 0 0 0.0\n'
            'HETATM 2 Co 2.0 0.0 0.0 Co 0 0 0.0\nEND\n')
        (self.source / 'trainset.in').write_text('ENERGY\n2 + cobalt/1 -20\nENDENERGY\n')
        (self.source / 'control').write_text('0 imetho\n')
        bundle = self.root / 'real-jax'
        bridge.convert(self.source, bundle, backend='jax-md', engine_python=os.environ['JAX_REAXFF_TEST_PYTHON'])
        workdir = self.candidate(bundle)
        try:
            fitness = bridge.evaluate(bundle, workdir, 'single')
        except bridge.InputError:
            self.fail((workdir / 'evaluator.stdout.log').read_text(errors='replace') + '\n' +
                      (workdir / 'evaluator.stderr.log').read_text(errors='replace'))
        response = json.loads((workdir / 'fitness.json').read_text())
        self.assertEqual(response['target_count'], 1)
        self.assertEqual(response['settings']['energy_minimization_steps'], 0)
        self.assertGreaterEqual(fitness, 0)
        self.assertEqual((workdir / 'ffield').read_bytes(), (self.source / 'ffield').read_bytes())


if __name__ == '__main__':
    unittest.main()
