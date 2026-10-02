"""Document-only transfer tests. Fixtures are transport tests, not training plans."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from plan_bundle import START, END, CHECKS, SCIENCE, PHASES, extract, pack, read_document, validate_payload


class PortablePlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.files=self.root/'source'; self.files.mkdir()
        # Reads a bundled config relative to its own workspace, not the author cwd.
        self.script="from pathlib import Path\nimport json,sys\np=Path(__file__).parent/'settings.json'\nprint(json.dumps({'phase':sys.argv[1], 'seed':json.loads(p.read_text())['seed']}))\n"
        (self.files/'worker.py').write_text(self.script,encoding='utf-8')
        (self.files/'settings.json').write_text('{"seed":17}',encoding='utf-8')
        self.spec={'schema_version':1,'plan_id':'transport-fixture','purpose':'Transport test only, not GPU/training verification',
                   'scientific_spec':{key:{'fixture':True} for key in SCIENCE},'external_inputs':[],
                   'target_checks':{key:'Run on actual target; pending in fixture' for key in CHECKS},
                   'entrypoints':{key:['{python}','worker.py',key] for key in PHASES}}
        self.spec_path=self.root/'spec.json'; self.spec_path.write_text(json.dumps(self.spec),encoding='utf-8')
        self.overview=self.root/'overview.txt'; self.overview.write_text('Document-only transport test. GPU unverified.\n',encoding='utf-8')
        self.doc=self.root/'plan.md'
        pack(self.spec_path,self.files,self.overview,self.doc)

    def tampered(self,payload):
        path=self.root/'tampered.md'
        path.write_text(START+'\n```json\n'+json.dumps(payload)+'\n```\n'+END,encoding='utf-8')
        return path

    def test_document_alone_reconstructs_and_runs_in_clean_directory(self):
        transferred=self.root/'transferred.md'; shutil.copy2(self.doc,transferred)
        shutil.rmtree(self.files)  # The author sources are unavailable on the receiver.
        self.spec_path.unlink(); self.overview.unlink(); self.doc.unlink()
        target=self.root/'receiver'
        result=extract(transferred,target)
        self.assertFalse(result['code_executed']); self.assertFalse(result['target_verified'])
        manifest=json.loads((target/'portable-plan-manifest.json').read_text())
        for phase in PHASES:
            argv=manifest['spec']['entrypoints'][phase]
            run=subprocess.run([sys.executable,*argv[1:]],cwd=target,capture_output=True,text=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(json.loads(run.stdout),{'phase':phase,'seed':17})

    def test_extraction_never_executes_embedded_script(self):
        payload,_=read_document(self.doc)
        data=b'raise RuntimeError("must never execute during extraction")\n'
        row=next(x for x in payload['files'] if x['path']=='worker.py')
        row.update(base64=base64.b64encode(data).decode(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        extract(self.tampered(payload),self.root/'unexecuted')

    def test_byte_exact_text_and_markdown_fence_roundtrip(self):
        special=b'UTF-8 text: \xed\x95\x9c\xea\xb8\x80\r\n```json\n{}\n```\n<!-- TRAIN_RECIPE_BUNDLE_V1 -->\n'
        (self.files/'special.txt').write_bytes(special)
        path=self.root/'special-plan.md';pack(self.spec_path,self.files,self.overview,path)
        extract(path,self.root/'special-output')
        self.assertEqual((self.root/'special-output/special.txt').read_bytes(),special)

    def test_corrupt_file_rejected_before_destination_creation(self):
        payload,_=read_document(self.doc); payload['files'][0]['sha256']='0'*64
        target=self.root/'corrupt-output'
        with self.assertRaises(ValueError): extract(self.tampered(payload),target)
        self.assertFalse(target.exists())

    def test_traversal_absolute_windows_and_nonportable_paths_rejected(self):
        original,_=read_document(self.doc)
        for name in ('.','portable-plan-manifest.json/note.txt','../evil.py','/evil.py','C:/evil.py','a\\b.py','CON.txt','a./b.py','a//b.py','a/../b.py','file?.py'):
            payload=copy.deepcopy(original);payload['files'][0]['path']=name
            with self.subTest(name=name),self.assertRaises(ValueError): validate_payload(payload)

    def test_duplicate_case_collision_and_file_directory_collision_rejected(self):
        original,_=read_document(self.doc)
        for name in ('worker.py','WORKER.py','worker.py/nested.txt'):
            payload=copy.deepcopy(original);extra=copy.deepcopy(next(x for x in payload['files'] if x['path']=='worker.py'))
            extra['path']=name;payload['files'].append(extra)
            with self.subTest(name=name),self.assertRaises(ValueError): validate_payload(payload)

    def test_existing_destination_and_document_are_not_overwritten(self):
        target=self.root/'existing';target.mkdir();(target/'keep.txt').write_text('keep')
        with self.assertRaises(ValueError):extract(self.doc,target)
        self.assertEqual((target/'keep.txt').read_text(),'keep')
        old=self.doc.read_bytes()
        with self.assertRaises(FileExistsError):pack(self.spec_path,self.files,self.overview,self.doc)
        self.assertEqual(self.doc.read_bytes(),old)

    def test_missing_phase_or_target_check_or_source_contract_rejected(self):
        original,_=read_document(self.doc)
        for field,key in (('entrypoints','resume'),('target_checks','controller_recovery'),('scientific_spec','data')):
            payload=copy.deepcopy(original);del payload['spec'][field][key]
            with self.subTest(field=field),self.assertRaises(ValueError):validate_payload(payload)

    def test_external_entrypoint_not_in_document_rejected(self):
        payload,_=read_document(self.doc);payload['spec']['entrypoints']['run']=['{python}','/author/missing.py']
        with self.assertRaises(ValueError):validate_payload(payload)

    def test_python_option_entrypoint_rejected_before_extract(self):
        payload,_=read_document(self.doc)
        row=next(x for x in payload['files'] if x['path']=='worker.py');row['path']='-c'
        payload['spec']['entrypoints']={key:['{python}','-c','raise RuntimeError("not a file")'] for key in PHASES}
        target=self.root/'option-output'
        with self.assertRaises(ValueError):extract(self.tampered(payload),target)
        self.assertFalse(target.exists())

    def test_directory_case_alias_rejected(self):
        payload,_=read_document(self.doc)
        for name in ('Code/first.py','code/second.py'):
            row=copy.deepcopy(payload['files'][0]);row['path']=name;payload['files'].append(row)
        with self.assertRaises(ValueError):validate_payload(payload)

    def test_duplicate_bundle_and_json_keys_rejected(self):
        text=self.doc.read_text(encoding='utf-8');bad=self.root/'duplicate.md'
        bad.write_text(text+text,encoding='utf-8')
        with self.assertRaises(ValueError):read_document(bad)
        bad.write_text(START+'\n```json\n{"spec":{},"spec":{},"files":[]}\n```\n'+END,encoding='utf-8')
        with self.assertRaises(ValueError):read_document(bad)

    def test_invalid_base64_and_binary_payload_rejected(self):
        original,_=read_document(self.doc)
        for encoded in ('not base64!',base64.b64encode(b'\x00\xff').decode()):
            payload=copy.deepcopy(original);payload['files'][0]['base64']=encoded
            with self.subTest(encoded=encoded),self.assertRaises(ValueError):validate_payload(payload)

    def test_cli_verify_is_explicitly_not_gpu_acceptance(self):
        tool=Path(__file__).with_name('plan_bundle.py')
        result=subprocess.run([sys.executable,str(tool),'verify',str(self.doc)],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        record=json.loads(result.stdout)
        self.assertTrue(record['transport_valid']);self.assertFalse(record['target_verified'])
        self.assertFalse(record['semantic_training_correctness_verified'])


if __name__=='__main__': unittest.main()
