"""Narrow read-only collector correction: preserve absent namespace origin as unknown."""
import hashlib
from pathlib import Path

prior=Path(__file__).resolve().with_name('inspect_once.py')
raw=prior.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='40ef9e93a5587418e7c0ce1d487a49d42b0ce5af6740b0cd376cfd1dd052153f'
source=raw.decode()
replacements=[
    ("dest=HERE/(host+'.json')", "dest=HERE/(host+'-r2.json')"),
    ("resolved=Path(p).resolve(strict=True)", "resolved=Path(p).resolve(strict=False)"),
    (" for mod in ('torch_neuronx','torch_mlir','nki'):", " module_names={'torch_neuronx','torch_mlir','nki'}\n for record in record_rows:\n  module_names.update(row[0].split('/')[0] for row in record['selected_root_init_rows'] if re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',row[0].split('/')[0]))\n for mapping in finder_mappings.values():\n  if isinstance(mapping,dict):module_names.update(k for k in mapping if isinstance(k,str) and re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',k))\n if len(module_names)>32:raise RuntimeError('MODULE_MAPPING_COUNT_BOUND')\n for mod in sorted(module_names):"),
    ("p=allowed_module_path(row['origin']);pin,ident=file_record(p);row['origin_pin']=pin;row['origin_identity']=ident", "p=allowed_module_path(row['origin'])\n    if not p.is_file():row['origin_missing_or_namespace_without_init']=True;continue\n    pin,ident=file_record(p);row['origin_pin']=pin;row['origin_identity']=ident"),
    ("'resolved_statically':bool(seen)", "'resolved_statically':any('origin_pin' in r or (r['origin'] is None and r['locations']) for r in seen)"),
]
for old,new in replacements:
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(Path(__file__).resolve()),'exec'))
