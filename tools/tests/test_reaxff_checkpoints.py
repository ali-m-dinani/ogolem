from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_reaxff_bridge as fixtures
import reaxff_bridge as bridge


class CheckpointTest(unittest.TestCase):
    setUp = fixtures.BridgeTest.setUp
    bundle = fixtures.BridgeTest.bundle
    configure_mock = fixtures.BridgeTest.configure_mock

    def evaluate(self, bundle, name, values):
        workdir = self.root / name
        workdir.mkdir()
        (workdir / 'input.params').write_text(bridge.ogolem_parameters(values))
        return bridge.evaluate(bundle, workdir, 'single')

    def test_best_and_latest_are_saved_after_each_evaluation(self):
        bundle = self.bundle()
        self.configure_mock(bundle)
        self.assertEqual(self.evaluate(bundle, 'first', [1, 1]), 3)
        first_state = json.loads((bundle / 'progress.json').read_text())
        self.assertEqual(first_state['completed_evaluations'], 1)
        self.assertTrue((bundle / first_state['best']['evaluation'] / 'parameters.params').exists())
        self.assertEqual(self.evaluate(bundle, 'better', [0.5, 0.5]), 1.5)
        self.assertEqual(self.evaluate(bundle, 'worse', [2, 2]), 9)
        state = json.loads((bundle / 'progress.json').read_text())
        self.assertEqual(state['completed_evaluations'], 3)
        self.assertEqual(state['best']['fitness'], 1.5)
        self.assertEqual(state['latest']['fitness'], 9)
        self.assertEqual(bridge.read_ogolem_parameters((bundle / 'best.params').read_text(), 2), [0.5, 0.5])
        self.assertEqual(bridge.read_ogolem_parameters((bundle / 'latest.params').read_text(), 2), [2, 2])
        self.assertEqual((bundle / 'best.ffield').read_bytes(),
                         (bundle / state['best']['evaluation'] / 'ffield').read_bytes())

    def test_atomic_replace_failure_keeps_previous_file(self):
        target = self.root / 'checkpoint.json'
        target.write_bytes(b'previous complete checkpoint')
        with patch.object(bridge.os, 'replace', side_effect=OSError('simulated interruption')):
            with self.assertRaises(OSError):
                bridge.atomic_write(target, b'new checkpoint')
        self.assertEqual(target.read_bytes(), b'previous complete checkpoint')
        self.assertEqual(list(self.root.glob('.checkpoint.json.*.tmp')), [])

    def test_recovery_after_interruption_between_archive_and_progress(self):
        bundle = self.bundle()
        self.configure_mock(bundle)
        self.evaluate(bundle, 'first', [1, 1])
        previous = (bundle / 'progress.json').read_bytes()
        with patch.object(bridge, 'update_checkpoints', side_effect=OSError('simulated interruption')):
            with self.assertRaises(OSError):
                self.evaluate(bundle, 'second', [0.5, 0.5])
        self.assertEqual((bundle / 'progress.json').read_bytes(), previous)
        self.assertFalse((self.root / 'second/output.params').exists())
        self.assertEqual(len(list((bundle / 'evaluations').glob('candidate_*'))), 2)
        state = bridge.refresh_checkpoints(bundle)
        self.assertEqual(state['completed_evaluations'], 2)
        self.assertEqual(state['best']['fitness'], 1.5)
        self.assertEqual(bridge.refresh_checkpoints(bundle)['completed_evaluations'], 2)

    def test_incomplete_archive_is_not_published(self):
        bundle = self.bundle()
        self.configure_mock(bundle)
        self.evaluate(bundle, 'first', [1, 1])
        copy_file = shutil.copyfile

        def interrupted_copy(source, destination):
            if Path(destination).parent.name.startswith('.pending_'):
                raise OSError('copy interrupted')
            return copy_file(source, destination)

        with patch.object(bridge.shutil, 'copyfile', side_effect=interrupted_copy):
            with self.assertRaises(OSError):
                self.evaluate(bundle, 'second', [0.5, 0.5])
        self.assertEqual(len(list((bundle / 'evaluations').glob('.pending_*'))), 1)
        self.assertEqual(bridge.refresh_checkpoints(bundle)['completed_evaluations'], 1)

    def test_failed_engine_does_not_change_previous_checkpoints(self):
        bundle = self.bundle()
        self.configure_mock(bundle)
        self.evaluate(bundle, 'first', [1, 1])
        previous = (bundle / 'progress.json').read_bytes()
        self.configure_mock(bundle, 'raise SystemExit(3)\n')
        with self.assertRaises(bridge.InputError):
            self.evaluate(bundle, 'second', [0.5, 0.5])
        self.assertEqual((bundle / 'progress.json').read_bytes(), previous)
        self.assertEqual(len(list((bundle / 'evaluations').glob('candidate_*'))), 1)

    def test_legacy_archives_can_be_recovered_without_running_engine(self):
        bundle = self.bundle()
        self.configure_mock(bundle)
        self.evaluate(bundle, 'first', [1, 1])
        archive = next((bundle / 'evaluations').glob('candidate_*'))
        (archive / 'checkpoint.json').unlink()
        (archive / 'parameters.params').unlink()
        (bundle / 'progress.json').unlink()
        state = bridge.refresh_checkpoints(bundle)
        self.assertEqual(state['best']['fitness'], 3)
        self.assertTrue((archive / 'parameters.params').exists())
        self.assertTrue((bundle / 'best.ffield').exists())

    def test_invalid_recovery_does_not_replace_progress(self):
        bundle = self.bundle()
        self.configure_mock(bundle)
        self.evaluate(bundle, 'first', [1, 1])
        previous = (bundle / 'progress.json').read_bytes()
        archive = next((bundle / 'evaluations').glob('candidate_*'))
        (archive / 'fitness.json').write_text('{"fitness": -1}')
        with self.assertRaises(bridge.InputError):
            bridge.refresh_checkpoints(bundle)
        self.assertEqual((bundle / 'progress.json').read_bytes(), previous)

    def test_separate_processes_preserve_global_best_and_count(self):
        bundle = self.bundle()
        records = []
        for index in range(6):
            archive = bundle / 'evaluations' / f'candidate_{index}'
            archive.mkdir(parents=True)
            shutil.copyfile(self.source / 'ffield', archive / 'ffield')
            (archive / 'parameters.params').write_text(bridge.ogolem_parameters([1, 1], index + 1))
            record = {'evaluation': archive.relative_to(bundle).as_posix(), 'fitness': index + 1,
                      'completed_at': f'2026-11-04T00:00:0{index}+00:00'}
            bridge.write_json(archive / 'checkpoint.json', record)
            records.append(archive / 'checkpoint.json')
        script = ('import json,sys;sys.path.insert(0,sys.argv[1]);import reaxff_bridge as bridge;'
                  'bridge.update_checkpoints(sys.argv[2],json.loads(bridge.read_text(sys.argv[3])))')

        def update(record):
            return subprocess.run([sys.executable, '-c', script, str(Path(bridge.__file__).parent),
                                   str(bundle), str(record)], capture_output=True, text=True, timeout=30)

        with ThreadPoolExecutor(max_workers=6) as workers:
            results = list(workers.map(update, records))
        for result in results:
            self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads((bundle / 'progress.json').read_text())
        self.assertEqual(state['completed_evaluations'], 6)
        self.assertEqual(state['best']['fitness'], 1)


if __name__ == '__main__':
    unittest.main()
