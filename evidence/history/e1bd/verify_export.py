"""Offline byte/member checker only. No model/helper import, host access or audit verdict."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(raw): return hashlib.sha256(raw).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest-sha256',required=True);p.add_argument('--compare-canonical',action='store_true');a=p.parse_args()
    h=Path(__file__).resolve().parent;mraw=(h/'EXPORT_MANIFEST.json').read_bytes();assert sha(mraw)==a.manifest_sha256
    m=json.loads(mraw);assert m['submission_authorized']is False and m['package_audit']=='PENDING'
    expected=set(m['files'])|{'EXPORT_MANIFEST.json'};actual={str(x.relative_to(h))for x in h.rglob('*')if x.is_file()}
    assert actual==expected,'EXTRA_OR_MISSING_FILE';assert not any(x.is_symlink()for x in h.rglob('*'))
    for name,rec in m['files'].items():
        assert not Path(name).is_absolute() and '..'not in Path(name).parts
        raw=(h/name).read_bytes();assert (sha(raw),len(raw))==(rec['sha256'],rec['bytes']),name
    assert sha((h/'train.py').read_bytes())==m['source_sha256']
    assert sha((h/'audited_parent.py').read_bytes())==m['parent_sha256']
    e=json.loads((h/'cold-evidence.json').read_bytes());assert e['source_sha256']==m['source_sha256'] and e['receipt_count']==m['receipt_count']
    count=0
    for row in json.loads((h/'RAW_RECEIPT_INDEX.json').read_bytes()):
        raw=(h/row['path']).read_bytes();assert sha(raw)==row['sha256'];container=json.loads(raw)
        assert container['raw_archive']['sha256']==row['archive_sha256']
        for name,member in container['members'].items():
            b=member['text'].encode();assert (sha(b),len(b))==(member['sha256'],member['bytes'])
            assert row['members'][name]==dict(sha256=member['sha256'],bytes=member['bytes']);count+=1
    refs=json.loads((h/'REFERENCES.json').read_bytes())['references']
    if a.compare_canonical:
        for ref in refs:
            p=Path(ref['path']);assert p.is_file()and not p.is_symlink();raw=p.read_bytes()
            assert (sha(raw),len(raw))==(ref['sha256'],ref['bytes']),str(p)
    print(json.dumps(dict(status='BYTE_AND_RAW_MEMBER_INTEGRITY_ONLY_NOT_AUDIT_ACCEPTANCE',file_count=len(actual),raw_members_checked=count,canonical_references_checked=len(refs)if a.compare_canonical else 0,host_calls=0,model_or_torch_execution=False,submission_authorized=False)))

if __name__=='__main__':main()
