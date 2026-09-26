"""Same all-outcome collector, new assigned f804 host/label observation binding."""
import ast,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
raw=(HERE/'retrieve_native201.py').read_bytes()
EXPECTED_NATIVE201_RETRIEVER_SHA='5d1430ded0a26c1722d536a216ed86cd9573dd7f6adf1925b4261f834ad03035'
assert sys.argv[1]=='trn22-runtimegroups-native201'
text=raw.decode()
for a,b in [("assert sys.argv[1]=='trn21-localconstants-native201'","assert sys.argv[1]=='trn22-runtimegroups-native201'"),
 ("{'trn21-localconstants-native201':'trn21'}","{'trn22-runtimegroups-native201':'trn22'}"),
 ('20260926-k2-native201-*/','20260926-k2-runtimegroups-*/')]:
 assert text.count(a)==1,a
 text=text.replace(a,b)
assert EXPECTED_NATIVE201_RETRIEVER_SHA is not None and hashlib.sha256(raw).hexdigest()==EXPECTED_NATIVE201_RETRIEVER_SHA
ast.parse(text)
exec(compile(text,str(HERE/'retrieve_native201.py'),'exec'),{'__file__':str(HERE/'retrieve_native201.py'),'__name__':'__main__'})
