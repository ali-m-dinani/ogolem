import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reaxff_bridge as bridge


def force_field_text():
    row = lambda count: ''.join(f'{1.0:10.4f}' for index in range(count)) + '\n'
    return ('Test force field\n2 ! general\n    1.0000 ! first\n    2.0000 ! second\n'
            '1 ! atoms\nheader\nheader\nheader\n H' + row(8) + row(8) * 3
            + '1 ! bonds\nheader\n 1 1' + row(8) + row(8)
            + '1 ! off-diagonal\n 1 1' + row(6)
            + '1 ! angles\n 1 1 1' + row(7)
            + '1 ! torsions\n 1 1 1 1' + row(7)
            + '1 ! hydrogen bonds\n 1 1 1' + row(4))


GEO = ('BIOGRF 200\nDESCRP molecule\n'
       'HETATM 1 H 0.0 0.0 0.0 H 0 0 0.0\n'
       'HETATM 2 H 1.0 0.0 0.0 H 0 0 0.0\n'
       'CRYSTX 10 11 12 90 90 90\nEND\n')


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for name, text in {'ffield': force_field_text(), 'params': '1 1 1 0.1 0.0 5.0\n2 1 9 0.1 0.0 5.0\n',
                           'geo': GEO, 'trainset.in': 'ENERGY\n0.5 + molecule/2 - molecule / 1 -1.0\nENDENERGY\n',
                           'control': '1 imetho\n', 'iopt': '4\n'}.items():
            (self.source / name).write_text(text, encoding='utf-8')

    def bundle(self):
        bundle = self.root / 'bundle'
        bridge.convert(self.source, bundle, iterations=0, population=2)
        return bundle

    def configure_mock(self, bundle, body=None):
        evaluator = self.root / 'mock_evaluator.py'
        evaluator.write_text(body or (
            'import json\nfrom pathlib import Path\n'
            'request = json.loads(Path("request.json").read_text())\n'
            'assert not Path("iopt").exists()\n'
            'fitness = sum(value**2 for value in request["parameter_values"]) + 1\n'
            'Path("fitness.json").write_text(json.dumps({"fitness": fitness}))\n'), encoding='utf-8')
        bridge.write_json(bundle / 'evaluator.json', {'command': [sys.executable, str(evaluator)], 'timeout_seconds': 20})

    def candidate(self, bundle):
        workdir = self.root / 'candidate'
        workdir.mkdir()
        shutil.copyfile(bundle / 'initial.params', workdir / 'input.params')
        return workdir

    def test_all_sections_and_multiline_field_mapping(self):
        original = force_field_text()
        force_field = bridge.ForceField(original)
        text = '\n'.join(f'{section} 1 {parameter} 0.1 0 5' for section, parameter in
                         [(1, 1), (2, 32), (3, 16), (4, 6), (5, 7), (6, 7), (7, 4)])
        parameters, _ = bridge.read_parameters(text)
        force_field.bind(parameters)
        self.assertEqual(force_field.render(parameters, [item.initial for item in parameters]), original)
        changed = bridge.ForceField(force_field.render(parameters, [2.5] * 7))
        for key, location in force_field.locations.items():
            expected = 2.5 if key in {item.key for item in parameters} else location[3]
            self.assertEqual(changed.locations[key][3], expected)

    def test_bounds_duplicates_and_missing_bounds_rejected(self):
        for text in ['1 1 1 0.1 4 0', '1 1 1 0.1', '1 1 1 0.1 0 4\n1 1 1 0.1 0 4']:
            with self.subTest(text=text), self.assertRaises(bridge.InputError):
                bridge.read_parameters(text)
        parameters, warnings = bridge.read_parameters('1 1 1 0.1 -0.03 -0.20', True)
        self.assertEqual((parameters[0].lower, parameters[0].upper), (-0.20, -0.03))
        self.assertEqual(len(warnings), 1)

    def test_invalid_atom_types_are_reported_and_rejected_before_conversion(self):
        invalid = force_field_text().replace('1 ! torsions\n 1 1 1 1', '1 ! torsions\n 1 1 1 57')
        (self.source / 'ffield').write_text(invalid)
        report = bridge.audit(self.source)
        self.assertEqual(report['invalid_atom_type_references'][0]['invalid_indices'], [57])
        with self.assertRaisesRegex(bridge.InputError, 'unsupported atom-type references'):
            self.bundle()
        self.assertFalse((self.root / 'bundle').exists())

    def test_torsion_endpoint_wildcards_are_valid_but_central_wildcards_are_not(self):
        valid = force_field_text().replace('1 ! torsions\n 1 1 1 1', '1 ! torsions\n 0 1 1 0')
        bridge.ForceField(valid).validate_topology()
        invalid = valid.replace('1 ! torsions\n 0 1 1 0', '1 ! torsions\n 0 0 1 0')
        with self.assertRaisesRegex(bridge.InputError, 'unsupported atom-type references'):
            bridge.ForceField(invalid).validate_topology()

    def test_existing_bundle_with_invalid_topology_is_rejected_even_with_matching_hash(self):
        bundle = self.bundle()
        path = bundle / 'inputs' / 'ffield'
        path.write_text(force_field_text().replace('1 ! torsions\n 1 1 1 1', '1 ! torsions\n 1 1 1 57'))
        manifest = json.loads((bundle / 'manifest.json').read_text())
        manifest['sha256']['ffield'] = bridge.hashlib.sha256(path.read_bytes()).hexdigest()
        bridge.write_json(bundle / 'manifest.json', manifest)
        with self.assertRaisesRegex(bridge.InputError, 'unsupported atom-type references'):
            bridge.load_bundle(bundle)

    def test_energy_divisors_weights_and_periodicity(self):
        geometries = bridge.read_geometries(GEO)
        self.assertEqual(geometries['molecule']['cell'], [10, 11, 12, 90, 90, 90])
        for expression in ['+ molecule/2 - molecule /1', 'molecule 2 - molecule 1']:
            target = bridge.read_training(f'ENERGY\n0.5 {expression} -1\nENDENERGY', geometries)[0]
            self.assertEqual([term['coefficient'] for term in target['terms']], [0.5, -1])
            self.assertEqual(target['weight'], 0.5)
            self.assertEqual(target['target'], -1)

    def test_other_legacy_training_sections(self):
        text = ('CHARGE\nmolecule 0.1 1 -0.2\nENDCHARGE\n'
                'HEATFO\nmolecule 2 -5\nENDHEATFO\n'
                'GEOMETRY\nmolecule 0.1 1 2 1.0\nmolecule 1 0.5\nENDGEOMETRY\n'
                'CELL PARAMETERS\nmolecule 0.1 a 10\nENDCELL PARAMETERS\n')
        self.assertEqual(len(bridge.read_training(text, bridge.read_geometries(GEO))), 5)

    def test_invalid_training_rejected_before_output_creation(self):
        (self.source / 'trainset.in').write_text('ENERGY\n1 + absent/1 0\nENDENERGY\n')
        with self.assertRaisesRegex(bridge.InputError, 'missing geometries'):
            self.bundle()
        self.assertFalse((self.root / 'bundle').exists())
        report = bridge.audit(self.source)
        self.assertEqual(report['missing_geometries'], ['absent'])

    def test_bundle_preserves_inputs_and_roundtrips_parameters(self):
        bundle = self.bundle()
        for path in self.source.iterdir():
            self.assertEqual(path.read_bytes(), (bundle / 'inputs' / path.name).read_bytes())
        _, _, parameters = bridge.load_bundle(bundle)
        values = bridge.read_ogolem_parameters(bridge.read_text(bundle / 'initial.params'), 2)
        rendered = bridge.ForceField(bridge.read_text(self.source / 'ffield')).render(parameters, values)
        self.assertEqual(rendered, bridge.read_text(self.source / 'ffield'))
        self.assertEqual(len((bundle / 'fit-borders.aux').read_text().splitlines()), 3)
        with self.assertRaisesRegex(bridge.InputError, 'already exists'):
            bridge.convert(self.source, bundle)

    def test_evaluation_uses_fitness_directly_and_archives_candidate(self):
        bundle = self.bundle()
        self.configure_mock(bundle)
        candidate = self.candidate(bundle)
        self.assertEqual(bridge.evaluate(bundle, candidate, 'locopt'), 3.0)
        self.assertIn('<PARAMETERFITNESS>\n3\n', (candidate / 'output.params').read_text())
        self.assertEqual(len(list((bundle / 'evaluations').iterdir())), 1)
        self.assertEqual(bridge.read_ogolem_parameters((candidate / 'output.params').read_text(), 2), [1.0, 1.0])

    def test_unconfigured_evaluator_and_modified_bundle_fail(self):
        bundle = self.bundle()
        with self.assertRaisesRegex(bridge.InputError, 'no ReaxFF evaluator'):
            bridge.evaluate(bundle, self.candidate(bundle), 'single')
        (bundle / 'inputs' / 'control').write_text('changed')
        with self.assertRaisesRegex(bridge.InputError, 'modified'):
            bridge.load_bundle(bundle)

    def test_failed_evaluation_does_not_publish_output(self):
        bundle = self.bundle()
        self.configure_mock(bundle, 'raise SystemExit(3)\n')
        candidate = self.candidate(bundle)
        with self.assertRaisesRegex(bridge.InputError, 'exit code 3'):
            bridge.evaluate(bundle, candidate, 'single')
        self.assertFalse((candidate / 'output.params').exists())

    def test_response_scoring_uses_inverse_squared_weights(self):
        targets = [{'target': 1.0, 'weight': 0.5}, {'target': -2.0, 'weight': 2.0}]
        self.assertEqual(bridge.response_fitness({'predictions': [2.0, 0.0]}, targets), 5.0)
        self.assertEqual(bridge.response_fitness({'fitness': 5.0}, targets), 5.0)
        for response in [{'predictions': [0.0]}, {'fitness': float('nan')},
                         {'fitness': -1}, {'fitness': True}, {'predictions': [1, float('inf')]},
                         {'fitness': 5, 'predictions': [2, 0]}]:
            with self.subTest(response=response), self.assertRaises(bridge.InputError):
                bridge.response_fitness(response, targets)

    def test_field_overflow_and_corrupt_parameter_vector_rejected(self):
        field = bridge.ForceField(force_field_text())
        parameters, _ = bridge.read_parameters('1 1 1 0.1 0 1e20')
        with self.assertRaisesRegex(bridge.InputError, 'does not fit'):
            field.render(parameters, [1e19])
        for values in [bridge.ogolem_parameters([float('nan')]), bridge.ogolem_parameters([1, 2])]:
            with self.assertRaises(bridge.InputError):
                bridge.read_ogolem_parameters(values, 1)

    def test_out_of_bounds_initial_value_is_never_silently_clipped(self):
        (self.source / 'params').write_text('1 1 1 0.1 2 3\n')
        with self.assertRaisesRegex(bridge.InputError, 'cannot seed'):
            bridge.convert(self.source, self.root / 'seeded', seed=True)
        bundle = self.bundle()
        self.assertEqual(bridge.read_ogolem_parameters((bundle / 'initial.params').read_text(), 1), [1])
        self.assertNotIn('ParamSeedingFolder', (bundle / 'fit.ogo').read_text())

    def test_ogolem_executes_generated_external_interface(self):
        ogolem_root = Path(__file__).resolve().parents[2]
        launcher = ogolem_root / 'ogolem.cmd'
        if os.name != 'nt' or not (ogolem_root / 'build/libs/ogolem-snapshot.jar').exists():
            self.skipTest('Built Windows OGolem required for integration smoke test')
        bundle = self.bundle()
        self.configure_mock(bundle)
        environment = os.environ.copy()
        environment['OGO_EXTCALLCMD'] = str(bundle / 'ogo_evaluator.cmd')
        environment['REAXFF_BRIDGE_PYTHON'] = sys.executable
        result = subprocess.run([str(launcher), '--adaptive', 'fit.ogo', '1', 'params=initial.params'],
                                cwd=bundle, env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Fitness of this parameter set is 3.0', result.stdout)
        self.assertTrue((bundle / 'evaluations').is_dir())


if __name__ == '__main__':
    unittest.main()
