"""Convert legacy ReaxFF inputs to OGolem's external adaptive interface."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


class InputError(ValueError):
    pass


def number(text):
    value = float(text.replace('D', 'E').replace('d', 'e'))
    if not math.isfinite(value):
        raise InputError(f'Non-finite number: {text}')
    return value


def data_part(line):
    return re.split(r'[!#]', line, maxsplit=1)[0].strip()


def read_text(path):
    return Path(path).read_bytes().decode('utf-8-sig')


@dataclass
class Parameter:
    section: int
    entry: int
    parameter: int
    interval: float
    lower: float
    upper: float
    source_line: int
    comment: str = ''
    initial: float = 0.0

    @property
    def key(self):
        return self.section, self.entry, self.parameter


def read_parameters(text, normalize_bounds=False):
    parameters, warnings, seen = [], [], set()
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = data_part(line).split()
        if not fields:
            continue
        if len(fields) != 6:
            raise InputError(f'params:{line_number}: require section, entry, parameter, interval, lower, upper; OGolem needs explicit bounds')
        section, entry, parameter = map(int, fields[:3])
        if section == 1 and parameter == 0:
            parameter = 1
        interval, lower, upper = map(number, fields[3:])
        if section not in range(1, 8) or entry < 1 or parameter < 1:
            raise InputError(f'params:{line_number}: invalid one-based parameter pointer')
        if lower > upper:
            if not normalize_bounds:
                raise InputError(f'params:{line_number}: reversed bounds {lower}, {upper}; use --normalize-bounds to explicitly sort endpoints')
            warnings.append(f'params:{line_number}: sorted reversed bounds {lower}, {upper}')
            lower, upper = upper, lower
        if lower == upper:
            raise InputError(f'params:{line_number}: equal bounds; remove fixed parameters from params')
        if interval < 0:
            raise InputError(f'params:{line_number}: negative search interval')
        item = Parameter(section, entry, parameter, interval, lower, upper, line_number,
                         line.partition('!')[2].strip())
        if item.key in seen:
            raise InputError(f'params:{line_number}: duplicate pointer {item.key}')
        seen.add(item.key)
        parameters.append(item)
    if not parameters:
        raise InputError('params contains no parameters')
    return parameters, warnings


class ForceField:
    """Index classic seven-section ffields without rewriting untouched text."""

    def __init__(self, text):
        self.lines = text.splitlines(keepends=True)
        self.locations = {}
        self.elements = []
        self.topology = []
        self.entry_tokens = {}
        cursor = 1
        layouts = [(1, 0, 0, [1]), (2, 3, 1, [8, 8, 8, 8]),
                   (3, 1, 2, [8, 8]), (4, 0, 2, [6]),
                   (5, 0, 3, [7]), (6, 0, 4, [7]), (7, 0, 3, [4])]
        for section, header_lines, prefix, sizes in layouts:
            while cursor < len(self.lines) and not data_part(self.lines[cursor]):
                cursor += 1
            if cursor >= len(self.lines):
                raise InputError(f'ffield: missing section {section}')
            try:
                count = int(data_part(self.lines[cursor]).split()[0])
            except ValueError as error:
                raise InputError(f'ffield:{cursor + 1}: expected section {section} count') from error
            if count < 0:
                raise InputError(f'ffield:{cursor + 1}: negative section count')
            cursor += 1 + header_lines
            for entry in range(1, count + 1):
                parameter = 1
                for offset, size in enumerate(sizes):
                    if cursor >= len(self.lines):
                        raise InputError(f'ffield: truncated section {section}, entry {entry}')
                    content = re.split(r'[!#]', self.lines[cursor], maxsplit=1)[0]
                    tokens = list(re.finditer(r'\S+', content))
                    skip = prefix if offset == 0 else 0
                    if len(tokens) != size + skip:
                        raise InputError(f'ffield:{cursor + 1}: expected {size + skip} fields in classic section {section}; extended formats are not supported')
                    if skip:
                        self.topology.append(tuple(token.group() for token in tokens[:skip]))
                        self.entry_tokens[section, entry] = tuple(token.group() for token in tokens[:skip])
                    if section == 2 and offset == 0:
                        self.elements.append(tokens[0].group())
                    elif skip:
                        for token in tokens[:skip]:
                            int(token.group())
                    for token in tokens[skip:]:
                        value = number(token.group())
                        self.locations[section, entry, parameter] = (cursor, token.start(), token.end(), value)
                        parameter += 1
                    cursor += 1
        if any(data_part(line) for line in self.lines[cursor:]):
            raise InputError('ffield: unsupported data after the seventh section')

    def topology_issues(self):
        issues = []
        for (section, entry), tokens in self.entry_tokens.items():
            if section == 2:
                continue
            indices = list(map(int, tokens))
            invalid = [value for offset, value in enumerate(indices)
                       if not (1 <= value <= len(self.elements) or
                               (section == 6 and offset in (0, 3) and value == 0))]
            if invalid:
                issues.append({'section': section, 'entry': entry,
                               'source_line': self.locations[section, entry, 1][0] + 1,
                               'atom_type_indices': indices, 'invalid_indices': invalid,
                               'atom_type_count': len(self.elements)})
        return issues

    def validate_topology(self):
        issues = self.topology_issues()
        if issues:
            details = '; '.join(f'line {item["source_line"]}, section {item["section"]}, entry {item["entry"]}: '
                                f'{item["atom_type_indices"]}' for item in issues)
            raise InputError(f'ffield defines {len(self.elements)} atom types but has unsupported atom-type references: '
                             f'{details}. Correct these references or explicitly resolve intentionally disabled terms; '
                             'the bridge will not guess replacements or silently omit interactions.')

    def bind(self, parameters):
        for parameter in parameters:
            if parameter.key not in self.locations:
                raise InputError(f'params:{parameter.source_line}: pointer {parameter.key} is outside ffield')
            parameter.initial = self.locations[parameter.key][3]

    def render(self, parameters, values):
        if len(values) != len(parameters):
            raise InputError(f'Expected {len(parameters)} candidate parameters, got {len(values)}')
        replacements = {}
        for parameter, value in zip(parameters, values):
            if not math.isfinite(value):
                raise InputError('Candidate parameters must be finite')
            line_index, start, end, initial = self.locations[parameter.key]
            if value == initial:
                continue
            original = self.lines[line_index][start:end]
            decimal = re.fullmatch(r'[+-]?\d*\.(\d+)', original)
            if decimal is None:
                raise InputError(f'Cannot safely patch non-fixed-decimal field {parameter.key}: {original}')
            value_text = f'{value:.{len(decimal.group(1))}f}'
            available_start = start
            line = self.lines[line_index]
            while available_start > 0 and line[available_start - 1] == ' ':
                available_start -= 1
            if available_start > 0:
                available_start += 1
            width = end - available_start
            if len(value_text) > width:
                raise InputError(f'Value {value} does not fit original ffield field {parameter.key}')
            replacements.setdefault(line_index, []).append((available_start, end, value_text.rjust(width)))
        lines = self.lines.copy()
        for line_index, changes in replacements.items():
            for start, end, replacement in sorted(changes, reverse=True):
                lines[line_index] = lines[line_index][:start] + replacement + lines[line_index][end:]
        return ''.join(lines)


def read_geometries(text):
    geometries, current = {}, None
    for line_number, line in enumerate(text.splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        tag = tokens[0].upper()
        if tag in ('BIOGRF', 'XTLGRF'):
            if current is not None:
                raise InputError(f'geo:{line_number}: missing END')
            current = {'name': None, 'atoms': [], 'cell': None, 'source_line': line_number}
        elif tag == 'DESCRP':
            if current is None or len(tokens) != 2 or current['name'] is not None:
                raise InputError(f'geo:{line_number}: invalid DESCRP')
            current['name'] = tokens[1]
        elif tag in ('ATOM', 'HETATM'):
            if current is None or len(tokens) < 6:
                raise InputError(f'geo:{line_number}: invalid atom record')
            try:
                coords = [number(line[start:start + 10].strip()) for start in (30, 40, 50)]
            except ValueError:
                coords = list(map(number, tokens[3:6]))
            current['atoms'].append({'id': int(tokens[1]), 'element': tokens[2], 'xyz': coords})
        elif tag == 'CRYSTX':
            if current is None or len(tokens) != 7:
                raise InputError(f'geo:{line_number}: invalid CRYSTX')
            current['cell'] = list(map(number, tokens[1:]))
        elif tag == 'END':
            if current is None or not current['name'] or not current['atoms']:
                raise InputError(f'geo:{line_number}: incomplete structure')
            name = current['name']
            if name in geometries:
                raise InputError(f'geo:{line_number}: duplicate DESCRP {name}')
            atom_ids = [atom['id'] for atom in current['atoms']]
            if len(set(atom_ids)) != len(atom_ids):
                raise InputError(f'geo:{line_number}: duplicate atom IDs in {name}')
            geometries[name] = current
            current = None
    if current is not None or not geometries:
        raise InputError('geo must contain complete BIOGRF/XTLGRF blocks ending in END')
    return geometries


SECTIONS = {'CHARGE': 'CHARGE', 'CHARGES': 'CHARGE', 'HEATFO': 'HEATFO',
            'GEOMETRY': 'GEOMETRY', 'CELL': 'CELL', 'CELL PARAMETERS': 'CELL',
            'CELL PARAMETER': 'CELL', 'ENERGY': 'ENERGY'}


def read_training(text, geometries, allow_missing=False):
    section, targets = None, []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = data_part(raw)
        if not line:
            continue
        label = ' '.join(line.upper().split())
        if label in SECTIONS:
            if section is not None:
                raise InputError(f'trainset:{line_number}: nested section')
            section = SECTIONS[label]
            continue
        if label.startswith('END'):
            closing = label[3:].strip()
            if section is None or SECTIONS.get(closing) != section:
                raise InputError(f'trainset:{line_number}: mismatched closing section')
            section = None
            continue
        if section is None:
            raise InputError(f'trainset:{line_number}: unsupported section or data outside section')
        fields = line.split()
        record = {'section': section, 'source_line': line_number, 'raw': raw}
        if section == 'ENERGY':
            weight = number(fields[0])
            target = number(fields[-1])
            expression = line[len(fields[0]):].strip()
            expression = expression[:expression.rfind(fields[-1])].strip()
            terms, cursor = [], 0
            pattern = re.compile(r'\s*([+-]?)\s*([^\s/+-]+)\s*/?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)')
            while cursor < len(expression):
                match = pattern.match(expression, cursor)
                if match is None:
                    raise InputError(f'trainset:{line_number}: invalid energy expression near {expression[cursor:]}')
                sign, name, divisor = match.groups()
                denominator = number(divisor)
                if denominator == 0:
                    raise InputError(f'trainset:{line_number}: zero energy divisor')
                terms.append({'name': name, 'coefficient': (-1 if sign == '-' else 1) / denominator})
                cursor = match.end()
            if not terms:
                raise InputError(f'trainset:{line_number}: empty energy expression')
            record['terms'] = terms
            names = [term['name'] for term in terms]
        else:
            name, weight, target = fields[0], number(fields[1]), number(fields[-1])
            names = [name]
            record['name'] = name
            identifiers = fields[2:-1]
            if section == 'HEATFO' and identifiers:
                raise InputError(f'trainset:{line_number}: HEATFO expects name, weight, target')
            if section == 'CELL':
                if len(identifiers) != 1 or identifiers[0].lower() not in ('a', 'b', 'c', 'alpha', 'beta', 'gamma'):
                    raise InputError(f'trainset:{line_number}: invalid cell component')
                record['component'] = identifiers[0].lower()
            elif section in ('CHARGE', 'GEOMETRY'):
                atoms = list(map(int, identifiers))
                if (section == 'CHARGE' and len(atoms) != 1) or (section == 'GEOMETRY' and len(atoms) not in (0, 2, 3, 4)):
                    raise InputError(f'trainset:{line_number}: invalid atom count for {section}')
                record['atoms'] = atoms
                if name in geometries and not set(atoms).issubset({atom['id'] for atom in geometries[name]['atoms']}):
                    raise InputError(f'trainset:{line_number}: atom ID missing from {name}')
        if weight <= 0:
            raise InputError(f'trainset:{line_number}: weight must be positive')
        missing = set(names) - geometries.keys()
        if missing and not allow_missing:
            raise InputError(f'trainset:{line_number}: missing geometries {sorted(missing)}')
        if missing:
            record['missing_geometries'] = sorted(missing)
        record.update(weight=weight, target=target)
        targets.append(record)
    if section is not None or not targets:
        raise InputError('trainset has an unclosed section or no targets')
    return targets


def ogolem_parameters(values, fitness=1e100):
    formatted = '\n'.join(f'{value:.17g}' for value in values)
    return (f'<PARAMETERSFOR>\nexternalcaller:{len(values)}D\n</PARAMETERSFOR>\n'
            f'<PARAMETERFITNESS>\n{fitness:.17g}\n</PARAMETERFITNESS>\n'
            f'<PARAMETERS>\n<ATOM>\nXX\n</ATOM>\n<VALUES>\n{formatted}\n</VALUES>\n</PARAMETERS>\n')


def read_ogolem_parameters(text, count):
    blocks = re.findall(r'<VALUES>\s*(.*?)\s*</VALUES>', text, re.S | re.I)
    atoms = re.findall(r'<ATOM>\s*(.*?)\s*</ATOM>', text, re.S | re.I)
    if len(blocks) != 1 or atoms != ['XX']:
        raise InputError('Expected one OGolem parameter block with atom key XX')
    values = [number(line.split()[0]) for line in blocks[0].splitlines() if line.strip()]
    if len(values) != count:
        raise InputError(f'Expected {count} parameters, got {len(values)}')
    return values


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def source_parameters(source, normalize_bounds=False):
    source = Path(source)
    if (source / 'params').exists():
        return read_parameters(read_text(source / 'params'), normalize_bounds)
    fields = [ForceField(read_text(source / name)) for name in ('ffield_bool', 'ffield_min', 'ffield_max')]
    template = ForceField(read_text(source / 'ffield'))
    mask, lower, upper = fields
    for field in (lower, upper):
        if field.locations.keys() != mask.locations.keys() or field.topology != mask.topology:
            raise InputError('ffield mask/min/max layouts must match one another')
    if mask.elements != template.elements:
        raise InputError('ffield mask atom ordering must match ffield')
    rows, mapping_warnings = [], []
    for key, location in mask.locations.items():
        if location[3] not in (0, 1):
            raise InputError(f'ffield_bool flag must be zero or one: {key}')
        if location[3] == 1:
            section, entry, parameter = key
            mapped_key = key
            mask_entries = {index: tokens for index, tokens in mask.entry_tokens.items() if index[0] == section}
            template_entries = {index: tokens for index, tokens in template.entry_tokens.items() if index[0] == section}
            if section != 1 and mask_entries != template_entries:
                matches = [index for index, tokens in template_entries.items() if tokens == mask.entry_tokens[section, entry]]
                if len(matches) != 1:
                    raise InputError(f'Mask entry {key} has missing or ambiguous topology in ffield')
                mapped_key = (*matches[0], parameter)
                if mapped_key != key:
                    mapping_warnings.append(f'Mapped mask pointer {key} to ffield pointer {mapped_key} by atom indices')
            if mapped_key not in template.locations:
                raise InputError(f'Mask parameter {key} is absent from ffield')
            rows.append(' '.join(map(str, (*mapped_key, 0.0, lower.locations[key][3], upper.locations[key][3]))))
    parameters, warnings = read_parameters('\n'.join(rows), normalize_bounds)
    return parameters, warnings + mapping_warnings


def audit(source):
    source = Path(source)
    parameters, warnings = source_parameters(source, normalize_bounds=True)
    force_field = ForceField(read_text(source / 'ffield'))
    force_field.bind(parameters)
    geometries = read_geometries(read_text(source / 'geo'))
    targets = read_training(read_text(source / 'trainset.in'), geometries, allow_missing=True)
    missing = sorted({name for target in targets for name in target.get('missing_geometries', [])})
    outside = [asdict(item) for item in parameters if not item.lower <= item.initial <= item.upper]
    return {'parameters': len(parameters), 'geometries': len(geometries), 'targets': len(targets),
            'invalid_atom_type_references': force_field.topology_issues(),
            'missing_geometries': missing, 'out_of_bounds_initial_parameters': outside,
            'warnings': warnings, 'affected_target_lines': [target['source_line'] for target in targets if target.get('missing_geometries')],
            'original_iopt': read_text(source / 'iopt').strip() if (source / 'iopt').exists() else None}


def convert(source, destination, normalize_bounds=False, iterations=None, population=None, seed=False,
            profile='basic', backend='custom', engine_python='{python}', ams_job_file=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise InputError(f'Output already exists: {destination}; choose a new directory')
    if profile not in ('basic', 's4-inspired') or backend not in ('custom', 'ams', 'jax-md'):
        raise InputError('Unknown algorithm profile or backend')
    iterations = iterations if iterations is not None else (110000 if profile == 's4-inspired' else 100)
    population = population if population is not None else (500 if profile == 's4-inspired' else 10)
    parameters, warnings = source_parameters(source, normalize_bounds)
    force_field = ForceField(read_text(source / 'ffield'))
    force_field.validate_topology()
    force_field.bind(parameters)
    geometries = read_geometries(read_text(source / 'geo'))
    targets = read_training(read_text(source / 'trainset.in'), geometries)
    unknown = {atom['element'] for geo in geometries.values() for atom in geo['atoms']} - set(force_field.elements)
    if unknown:
        raise InputError(f'Geometry elements absent from ffield: {sorted(unknown)}')
    for parameter in parameters:
        if not parameter.lower <= parameter.initial <= parameter.upper:
            message = f'Initial value {parameter.initial} for {parameter.key} is outside [{parameter.lower}, {parameter.upper}]'
            if seed:
                raise InputError(message + '; cannot seed an out-of-bounds candidate')
            warnings.append(message + '; original preserved, population initializes within bounds')
    if iterations < 0 or population < 2:
        raise InputError('iterations must be nonnegative and population must be at least 2')
    if profile == 's4-inspired' and len(parameters) < 3:
        raise InputError('S4 arctic crossover requires at least three selected parameters')
    if ams_job_file is not None:
        if backend != 'ams':
            raise InputError('--ams-job-file requires --backend ams')
        read_text(ams_job_file)
    for optional in ('control', 'iopt'):
        if (source / optional).exists():
            read_text(source / optional)
    destination.mkdir(parents=True)
    inputs = destination / 'inputs'
    inputs.mkdir()
    hashes = {}
    for name in ('ffield', 'params', 'geo', 'trainset.in', 'control', 'iopt', 'ffield_bool', 'ffield_min', 'ffield_max'):
        if (source / name).exists():
            shutil.copyfile(source / name, inputs / name)
            hashes[name] = hashlib.sha256((inputs / name).read_bytes()).hexdigest()
    if 'params' not in hashes:
        (inputs / 'params').write_text(''.join(
            f'{item.section} {item.entry} {item.parameter} {item.interval:.17g} {item.lower:.17g} {item.upper:.17g}\n'
            for item in parameters), encoding='utf-8')
        hashes['params'] = hashlib.sha256((inputs / 'params').read_bytes()).hexdigest()
    manifest = {'format_version': 1, 'parameters': [asdict(item) for item in parameters],
                'geometries': geometries, 'targets': targets, 'warnings': warnings,
                'sha256': hashes, 'fitness': 'sum(((prediction - target) / weight)**2)',
                'seeded': seed, 'profile': profile, 'backend': backend}
    write_json(destination / 'manifest.json', manifest)
    write_json(destination / 'evaluator.json', {'command': [], 'timeout_seconds': 3600,
               'note': 'Configure an argv list for a trusted evaluation-only adapter. It runs in the candidate directory and must write fitness.json. Do not use an optimization job.'})
    (destination / 'fit-borders.aux').write_text('###OGOLEMAUX###\n' + ''.join(
        f'{item.lower:.17g} {item.upper:.17g}\n' for item in parameters), encoding='utf-8')
    initial = ogolem_parameters([item.initial for item in parameters])
    (destination / 'initial.params').write_text(initial, encoding='utf-8')
    seed_line = ''
    if seed:
        (destination / 'seeds').mkdir()
        (destination / 'seeds' / 'initial.params').write_text(initial, encoding='utf-8')
        seed_line = 'ParamSeedingFolder=seeds\n'
    (destination / 'dummy.xyz').write_text('1\nExternal fitness placeholder, not a training geometry\nH 0.0 0.0 0.0\n', encoding='utf-8')
    operators = 'ParameterGlobOpt=parameters{xover(portugal:nocuts=1)mutation(germany:)}\n'
    settings = ''
    if profile == 's4-inspired':
        operators = ('ParameterGlobOpt=parameters{xover(multiple:20%arctic:nomix=3,order=2,random|'
                     '80%portugal:nocuts=1)mutation(multiple:20%germany:mode=all|'
                     '40%arctic:mode=multi,submode=current,shape=0.01|'
                     '40%arctic:mode=multi,submode=current,shape=0.8)}\n')
        settings = ('CrossProbability=0.8\nMutationRatio=0.2\nParentsChoice=fitnessrankbased:gausswidth=0.3\n'
                    'WhichDiversityCheck=0\nParamEnergyDiv=0\nParamThreshDiv=0\n'
                    'DoNiching=true\nWhichNicher=0\nNichesPerDim=30\nMaxIndividualsPerNiche=2\n'
                    'ParamSerializeAfterNewBest=true\n')
    (destination / 'fit.ogo').write_text(
        '###OGOLEM###\n<GEOMETRY>\nNumberOfParticles=1\n<MOLECULE>\nH;0.0;0.0;0.0\n</MOLECULE>\n</GEOMETRY>\n'
        f'<ADAPTIVE>\nAdaptivableChoice=externalcaller:{len(parameters)}D\n'
        + operators + settings +
        f'ParamGlobOptIter={iterations}\nPopulationSize={population}\n'
        'ParameterLocOpt=external:backend=alldefault\nAcceptableFitness=0.0\n'
        + seed_line + '<REFERENCE>\n<ENERGY>\n0.0\n</ENERGY>\n<PATH>\ndummy.xyz\n</PATH>\n</REFERENCE>\n</ADAPTIVE>\n', encoding='utf-8')
    shutil.copyfile(__file__, destination / 'bridge.py')
    (destination / 'ogo_evaluator.cmd').write_text(
        '@echo off\n"%REAXFF_BRIDGE_PYTHON%" "%~dp0bridge.py" evaluate --bundle "%~dp0." --mode %1\nexit /b %ERRORLEVEL%\n', encoding='utf-8')
    (destination / 'ogo_evaluator.sh').write_bytes(
        b'#!/bin/sh\nexec "${REAXFF_BRIDGE_PYTHON:-python3}" "$(dirname "$0")/bridge.py" evaluate --bundle "$(dirname "$0")" --mode "$1"\n')
    if os.name != 'nt':
        (destination / 'ogo_evaluator.sh').chmod(0o755)
    from reaxff_engines import configure_backend
    configure_backend(destination, backend, engine_python, ams_job_file)
    return manifest


def load_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads(read_text(bundle / 'manifest.json'))
    if manifest['format_version'] != 1:
        raise InputError('Unsupported bundle version')
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((bundle / 'inputs' / name).read_bytes()).hexdigest() != expected:
            raise InputError(f'Bundle input was modified: {name}; regenerate the bundle')
    ForceField(read_text(bundle / 'inputs' / 'ffield')).validate_topology()
    return bundle, manifest, [Parameter(**item) for item in manifest['parameters']]


def evaluator_config(bundle):
    config = json.loads(read_text(Path(bundle) / 'evaluator.json'))
    command = config.get('command')
    if not isinstance(command, list) or not command or not all(isinstance(arg, str) and arg for arg in command):
        raise InputError('Configure evaluator.json command as a nonempty argv list; no ReaxFF evaluator has been selected')
    timeout = config.get('timeout_seconds', 3600)
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise InputError('timeout_seconds must be finite and positive')
    command = [arg.replace('{python}', sys.executable).replace('{bundle}', str(Path(bundle).resolve())) for arg in command]
    return command, timeout


def response_fitness(response, targets):
    if not isinstance(response, dict) or ('fitness' in response) == ('predictions' in response):
        raise InputError('fitness.json must contain exactly one of fitness or predictions')
    if 'predictions' in response:
        predictions = response['predictions']
        if not isinstance(predictions, list) or len(predictions) != len(targets):
            raise InputError('predictions must have one value per manifest target in the same order')
        errors = []
        for prediction, target in zip(predictions, targets):
            if isinstance(prediction, bool) or not isinstance(prediction, (int, float)) or not math.isfinite(prediction):
                raise InputError('Predictions must be finite numbers')
            residual = (prediction - target['target']) / target['weight']
            errors.append(residual * residual)
        fitness = math.fsum(errors)
    else:
        fitness = response['fitness']
    if isinstance(fitness, bool) or not isinstance(fitness, (int, float)) or not math.isfinite(fitness) or fitness < 0:
        raise InputError('fitness.json must yield a finite, nonnegative numeric fitness')
    return fitness


def atomic_write(path, content):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def checkpoint_lock(bundle, timeout=60):
    with (Path(bundle) / '.checkpoint.lock').open('a+b') as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b'0')
            stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise InputError('Timed out waiting for checkpoint writer; candidate archive is preserved')
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_checkpoint_state(bundle, state):
    for label in ('best', 'latest'):
        archive = bundle / state[label]['evaluation']
        atomic_write(bundle / f'{label}.ffield', (archive / 'ffield').read_bytes())
        atomic_write(bundle / f'{label}.params', (archive / 'parameters.params').read_bytes())
    atomic_write(bundle / 'progress.json', (json.dumps(state, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def update_checkpoints(bundle, record):
    bundle = Path(bundle)
    with checkpoint_lock(bundle):
        progress = bundle / 'progress.json'
        if progress.exists():
            state = json.loads(read_text(progress))
            best = state['best'] if state['best']['fitness'] <= record['fitness'] else record
            count = state['completed_evaluations'] + 1
        else:
            best, count = record, 1
        state = {'format_version': 1, 'completed_evaluations': count, 'best': best, 'latest': record}
        write_checkpoint_state(bundle, state)
    return state


def refresh_checkpoints(bundle):
    bundle, manifest, parameters = load_bundle(bundle)
    force_field = ForceField(read_text(bundle / 'inputs' / 'ffield'))
    records = []
    with checkpoint_lock(bundle):
        for archive in sorted((bundle / 'evaluations').glob('candidate_*')):
            if not archive.is_dir():
                continue
            try:
                response = json.loads(read_text(archive / 'fitness.json'))
                fitness = response_fitness(response, manifest['targets'])
                request = json.loads(read_text(archive / 'request.json'))
                values = request['parameter_values']
                values = read_ogolem_parameters(ogolem_parameters(values, fitness), len(parameters))
                if any(not item.lower <= value <= item.upper for item, value in zip(parameters, values)):
                    raise InputError('Archived parameters are outside bounds')
                if (archive / 'ffield').read_bytes() != force_field.render(parameters, values).encode('utf-8'):
                    raise InputError('Archived force field does not match its parameter vector')
                completed = (archive / 'fitness.json').stat().st_mtime_ns
                record = {'evaluation': archive.relative_to(bundle).as_posix(), 'fitness': fitness,
                          'completed_at': datetime.fromtimestamp(completed / 1e9, timezone.utc).isoformat()}
                if (archive / 'checkpoint.json').exists():
                    metadata = json.loads(read_text(archive / 'checkpoint.json'))
                    record['completed_at'] = metadata['completed_at']
                atomic_write(archive / 'parameters.params', ogolem_parameters(values, fitness).encode('utf-8'))
                records.append(record)
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise InputError(f'Cannot recover checkpoint from {archive}: {error}') from error
        if not records:
            raise InputError('No completed evaluation archives found; existing checkpoints were not changed')
        records.sort(key=lambda item: item['completed_at'])
        state = {'format_version': 1, 'completed_evaluations': len(records),
                 'best': min(records, key=lambda item: item['fitness']), 'latest': records[-1]}
        write_checkpoint_state(bundle, state)
    return state


def evaluate(bundle, workdir, mode):
    bundle, manifest, parameters = load_bundle(bundle)
    workdir = Path(workdir).resolve()
    command, timeout = evaluator_config(bundle)
    values = read_ogolem_parameters(read_text(workdir / 'input.params'), len(parameters))
    for parameter, value in zip(parameters, values):
        if not parameter.lower <= value <= parameter.upper:
            raise InputError(f'Candidate value outside bounds: {parameter.key} = {value}')
    for name in ('output.params', 'fitness.json'):
        if (workdir / name).exists():
            raise InputError(f'Refusing stale evaluation output: {workdir / name}')
    force_field = ForceField(read_text(bundle / 'inputs' / 'ffield'))
    candidate = force_field.render(parameters, values)
    for name in manifest['sha256']:
        if name not in ('ffield', 'iopt'):
            shutil.copyfile(bundle / 'inputs' / name, workdir / name)
    (workdir / 'ffield').write_bytes(candidate.encode('utf-8'))
    write_json(workdir / 'request.json', {'mode': 'evaluate', 'ogolem_mode': mode,
               'manifest': str(bundle / 'manifest.json'), 'parameter_values': values,
               'note': 'Return total fitness in fitness.json. The adapter must select engine-specific evaluation settings; original iopt is intentionally not copied.'})
    with (workdir / 'evaluator.stdout.log').open('wb') as stdout, (workdir / 'evaluator.stderr.log').open('wb') as stderr:
        result = subprocess.run(command, cwd=workdir, stdout=stdout, stderr=stderr,
                                timeout=timeout, check=False)
    if result.returncode:
        raise InputError(f'Evaluator failed with exit code {result.returncode}; see {workdir}')
    response = json.loads(read_text(workdir / 'fitness.json'))
    fitness = response_fitness(response, manifest['targets'])
    if (workdir / 'ffield').read_bytes() != candidate.encode('utf-8'):
        raise InputError('Evaluator changed ffield; an evaluation-only adapter is required')
    archive = bundle / 'evaluations'
    archive.mkdir(exist_ok=True)
    evaluation_dir = Path(tempfile.mkdtemp(prefix='.pending_', dir=archive))
    for name in ('ffield', 'fitness.json', 'request.json', 'evaluator.stdout.log', 'evaluator.stderr.log',
                 'control', 'params', 'iopt', 'fort.99', 'engine.stdout.log', 'engine.stderr.log'):
        if (workdir / name).is_file():
            shutil.copyfile(workdir / name, evaluation_dir / name)
    if (workdir / 'jax-output').is_dir():
        shutil.copytree(workdir / 'jax-output', evaluation_dir / 'jax-output')
    completed_dir = archive / ('candidate_' + evaluation_dir.name.removeprefix('.pending_'))
    record = {'evaluation': completed_dir.relative_to(bundle).as_posix(), 'fitness': fitness,
              'completed_at': datetime.now(timezone.utc).isoformat()}
    parameter_output = ogolem_parameters(values, fitness).encode('utf-8')
    atomic_write(evaluation_dir / 'parameters.params', parameter_output)
    atomic_write(evaluation_dir / 'checkpoint.json', (json.dumps(record, indent=2) + '\n').encode('utf-8'))
    for name in ('ffield', 'fitness.json', 'request.json'):
        with (evaluation_dir / name).open('r+b') as stream:
            os.fsync(stream.fileno())
    os.replace(evaluation_dir, completed_dir)
    update_checkpoints(bundle, record)
    atomic_write(workdir / 'output.params', parameter_output)
    return fitness


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='action', required=True)
    inspection = subcommands.add_parser('inspect')
    inspection.add_argument('source', type=Path)
    inspection.add_argument('--report', type=Path)
    conversion = subcommands.add_parser('convert')
    conversion.add_argument('source', type=Path)
    conversion.add_argument('destination', type=Path)
    conversion.add_argument('--normalize-bounds', action='store_true')
    conversion.add_argument('--seed', action='store_true')
    conversion.add_argument('--iterations', type=int)
    conversion.add_argument('--population', type=int)
    conversion.add_argument('--profile', choices=('basic', 's4-inspired'), default='basic')
    conversion.add_argument('--backend', choices=('custom', 'ams', 'jax-md'), default='custom')
    conversion.add_argument('--engine-python', default='{python}')
    conversion.add_argument('--ams-job-file', type=Path)
    reconstruction = subcommands.add_parser('render')
    reconstruction.add_argument('--bundle', type=Path, required=True)
    reconstruction.add_argument('--parameters', type=Path, required=True)
    reconstruction.add_argument('--output', type=Path, required=True)
    check = subcommands.add_parser('check')
    check.add_argument('--bundle', type=Path, required=True)
    recovery = subcommands.add_parser('refresh-checkpoints')
    recovery.add_argument('--bundle', type=Path, required=True)
    evaluation = subcommands.add_parser('evaluate')
    evaluation.add_argument('--bundle', type=Path, required=True)
    evaluation.add_argument('--workdir', type=Path, default=Path.cwd())
    evaluation.add_argument('--mode', choices=('single', 'locopt'), default='single')
    args = parser.parse_args(argv)
    try:
        if args.action == 'inspect':
            report = audit(args.source)
            if args.report:
                if args.report.exists():
                    raise InputError('Report already exists; choose a new path')
                write_json(args.report, report)
            print(json.dumps(report, indent=2))
        elif args.action == 'convert':
            manifest = convert(args.source, args.destination, args.normalize_bounds,
                               args.iterations, args.population, args.seed, args.profile, args.backend,
                               args.engine_python, args.ams_job_file)
            print(f'Converted {len(manifest["parameters"])} parameters, {len(manifest["geometries"])} geometries, {len(manifest["targets"])} targets')
            for warning in manifest['warnings']:
                print('WARNING:', warning)
            if args.backend == 'custom':
                print('Configure evaluator.json before running OGolem; see tools/REAXFF_BRIDGE.md.')
            else:
                print(f'{args.backend} adapter configured. Review engine.json and validate the engine before fitting; see tools/REAXFF_BRIDGE.md.')
        elif args.action == 'render':
            bundle, manifest, parameters = load_bundle(args.bundle)
            values = read_ogolem_parameters(read_text(args.parameters), len(parameters))
            if args.output.exists():
                raise InputError('Output already exists; choose a new path')
            args.output.write_bytes(ForceField(read_text(bundle / 'inputs' / 'ffield')).render(parameters, values).encode('utf-8'))
        elif args.action == 'refresh-checkpoints':
            print(json.dumps(refresh_checkpoints(args.bundle), indent=2))
        elif args.action == 'check':
            bundle, manifest, parameters = load_bundle(args.bundle)
            evaluator_config(bundle)
            print(f'Bundle and evaluator configuration validated: {len(parameters)} parameters; engine execution not tested')
        else:
            print('Fitness:', evaluate(args.bundle, args.workdir, args.mode))
    except (ValueError, OSError, KeyError, IndexError, OverflowError, subprocess.SubprocessError) as error:
        parser.exit(2, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
