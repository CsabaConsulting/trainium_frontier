"""Literal choice-only rebind of the existing bounded native launch observer."""
import ast,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
old=HERE/'observe_k2_launch.py';raw=old.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='cb10f7ed5ff63eb88dd36ce0251e04111de2debdd1eb98515dc8825ca97b14f3'
text=raw.decode();assert text.count("['trn22-on-parentcontrol']")==1
text=text.replace("['trn22-on-parentcontrol']","['trn22-indexrepair']")
ast.parse(text)
exec(compile(text,str(old),'exec'),{'__file__':str(old),'__name__':'__main__'})
