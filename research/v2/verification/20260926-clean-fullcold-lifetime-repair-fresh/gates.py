import ast, fcntl, hashlib, importlib.util, json, os, pathlib, shlex, sys, types
sys.dont_write_bytecode=True
P=pathlib.Path('/home/poojith-devan/clean-fullcold-lifetime-fresh-TOCGAp')
O=pathlib.Path(__file__).parent
R=P/'new/research/v2/proposals/20260926-confirmed36-clean-cold-full-trn22-r2'
def check(v,m):
    if not v:raise RuntimeError(m)
b=json.loads((R/'NATIVE_BINDING.disabled.json').read_text())
for name,key in [('host_driver.py','driver_sha256'),('lifetime_guard.py','lifetime_guard_sha256'),('package/launch-command.txt','package_launch_sha256'),('package/provenance.json','package_provenance_sha256')]:
    check(hashlib.sha256((R/name).read_bytes()).hexdigest()==b[key],name)
check(shlex.split((R/'package/launch-command.txt').read_text())[1:]==b['training_command'],'long argv')
tree=ast.parse((R/'host_driver.py').read_text())
pre=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight')
gates=[n for n in pre.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='minimum_gib' for t in n.targets)) or (isinstance(n,ast.Assert) and isinstance(n.msg,ast.Constant) and n.msg.value=='DISK_HEADROOM')]
check(len(gates)==2,'disk statements')
code=compile(ast.Module(body=gates,type_ignores=[]),'actual-preflight-disk-gates','exec',optimize=0)
train={};ev={};cases=[]
for env,floor in ((train,13.5),(ev,10.5)):
    for difference in (-1,0,1):
        scope={'TRAIN_ENV':train,'env':env,'v':types.SimpleNamespace(f_bavail=int(floor*1024**3)+difference,f_frsize=1)}
        try:exec(code,scope);passed=True
        except AssertionError:passed=False
        check(passed==(difference>=0),'disk boundary');cases.append([floor,difference,passed])
spec=importlib.util.spec_from_file_location('g',R/'lifetime_guard.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
lock=O/('identity-'+str(sys.flags.optimize));lock.touch();fd=os.open(lock,os.O_RDWR);st=os.fstat(fd)
try:
    try:g.require_original_lock(fd,lock,(st.st_dev,st.st_ino));raise RuntimeError('accepted unlocked')
    except RuntimeError as e:check(str(e)=='ORIGINAL_LOCK_NOT_HELD','unexpected unlocked error')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);g.require_original_lock(fd,lock,(st.st_dev,st.st_ino))
    try:g.require_original_lock(fd,lock,(st.st_dev,st.st_ino+1));raise RuntimeError('accepted wrong inode')
    except RuntimeError as e:check(str(e)=='ORIGINAL_LOCK_IDENTITY_CHANGED','unexpected inode error')
finally:os.close(fd)
check('(66305,832589)' in (R/'host_driver.py').read_text(),'final original identity missing')
check('OPTIMIZED_DRIVER_FORBIDDEN' in (R/'host_driver.py').read_text(),'optimized guard missing')
result={'disk_exact_boundaries':cases,'actual_lock_checks':'PASS','package_science_command_and_binding_pins':'PASS','optimized_verifier':bool(sys.flags.optimize)}
(O/('gates-'+str(sys.flags.optimize)+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
