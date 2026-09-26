"""Export safety/rendering tests only; synthetic metrics are never native proof."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit_export',HERE/'export_cold_audit.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class AuditExport(unittest.TestCase):
    def test_missing_actual_cold_never_runs_analyzer_or_exports(self):
        real=Path.is_file
        target=m.OP/'manual-jobs/trn21-finalcold/results-bundle.json'
        with patch.object(Path,'is_file',lambda p:False if p==target else real(p)), \
             patch.object(m.subprocess,'run',side_effect=AssertionError('NO_ANALYZER_BEFORE_BUNDLE')), \
             patch.object(Path,'mkdir',side_effect=AssertionError('NO_EXPORT')):
            with self.assertRaisesRegex(ValueError,'ACTUAL_COLD_BUNDLE_MISSING_NO_EXPORT'):
                m.verified_inputs()

    def fixture(self):
        source=(HERE/'train.py').read_bytes()
        parent=(HERE.parent/'confirmed-clock36-116188/train.py').read_bytes()
        cmd=(HERE/'launch-command.txt').read_bytes()
        names=('operator-preflight.json','NATIVE_BINDING.json','terminal.json',
               'train/rehearsal.json','eval/rehearsal.json','fresh-public20m.log',
               'checkpoint.json','checkpoint-after-postflight.json')
        bundle={'files':{n:{'text':'SYNTHETIC_TEST_ONLY'} for n in names},'postflight':{'test_only':True}}
        row={'public_20m':1.234567,'steps':123,'cold_startup_seconds':901.0,
             'startup_excess_seconds':1.0,'charged_through_exit_seconds':1790.0,
             'raw_archive_sha256':'synthetic-test-only'}
        return source,parent,cmd,bundle,row

    def test_five_attachments_exact_source_and_truthful_scope(self):
        values=self.fixture()
        files=m.build_files(*values)
        self.assertEqual(set(files),{'train.py','accepted_parent.py','README.md','AUDIT_BRIEF.md','cold-evidence.json'})
        self.assertEqual(files['train.py'],values[0]);self.assertEqual(files['accepted_parent.py'],values[1])
        brief=files['AUDIT_BRIEF.md'].decode()
        self.assertIn(values[2].decode().strip(),brief)
        self.assertIn('1.234567000000',brief)
        self.assertIn('NOT upload approval',brief)
        evidence=json.loads(files['cold-evidence.json'])
        self.assertIsNone(evidence['new_official_score'])
        self.assertEqual(evidence['status'],'AUDIT_HANDOFF_NOT_UPLOAD_APPROVAL')

    def test_mismatched_source_refuses_render(self):
        values=list(self.fixture());values[0]+=b'\n'
        with self.assertRaisesRegex(ValueError,'CPU_RECEIPT_SOURCE_BINDING_DRIFT'):
            m.build_files(*values)


if __name__=='__main__':unittest.main()
