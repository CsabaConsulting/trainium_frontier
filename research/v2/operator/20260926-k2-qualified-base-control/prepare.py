"""Disabled mechanical control-role rebind; no new source, host or model work."""
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
def sha(x):return hashlib.sha256(x).hexdigest()
proposal_path='research/v2/proposals/20260926-k2-qualified-base-control/EXPERIMENT_PROPOSAL.json'
proposal_raw=(REPO/proposal_path).read_bytes()
assert sha(proposal_raw)=='548bd3bcacd3e8746bce2b908021776247e6607d37b2872920feafe9c7572a28'
proposal=json.loads(proposal_raw)
source_sha=proposal['candidate']['source_sha256']
parent_control_on=json.loads((HERE.parent/'20260926-k2-sprint-timing/trn22-on-parentcontrol-prepared.json').read_text())['binding']
old_sha={'trn21':'a9d360d383e28f0b085069901d7b2c73dd6e4d4e79a04f6eb28588beaf8856c8','trn22':'0b55c9258af5d39a441eb34f11ef6bf260b0a81ef8e5aa23383b924e9533df44'}
rows=[]
for host in ('trn21','trn22'):
    raw=(HERE.parent/'20260926-k2-sprint-full'/(host+'-prepared.json')).read_bytes()
    assert sha(raw)==old_sha[host]
    original=json.loads(raw);binding=original['binding']
    files={n:base64.b64decode(v['b64']) for n,v in original['files'].items()}
    for n,v in files.items():assert sha(v)==original['files'][n]['sha256']
    assert sha(files['package/train.py'])==binding['source_sha256']==source_sha
    assert sha(files['package/launch-command.txt'])==proposal['candidate']['launch_command_sha256']
    assert binding['launch_command']==proposal['candidate']['launch_command']
    assert binding['label']==next(x['label'] for x in proposal['host_alternatives'] if x['host']==host)
    oldprov=files['package/provenance.json'];prov=json.loads(oldprov)
    prov.update(base_sha256=source_sha,kind='control',arm_id='K2-qualified-base-control',
                run_purpose='quality',prospective_control_role_frozen_before_launch=True,
                no_duplicate_throughput_arm_record=True)
    driver=files['host_driver.py'].decode()
    if host=='trn22':
        old_owner=binding['cache_owner_label'];new_owner=parent_control_on['label']
        assert old_owner=='fs-20260926-k2-sprint-on-trn22-r1'
        assert new_owner=='fs-20260926-k2-sprint-on-parentcontrol-trn22-r1'
        driver=driver.replace(old_owner,new_owner)
        binding['train_environment']={k:(v.replace(old_owner,new_owner) if isinstance(v,str) else v) for k,v in binding['train_environment'].items()}
        binding['cache_paths_must_already_exist']=[p.replace(old_owner,new_owner) for p in binding['cache_paths_must_already_exist']]
        binding['cache_owner_label']=new_owner;prov['cache_owner_label']=new_owner
        for key in ('NEURON_COMPILE_CACHE_URL','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):
            assert binding['train_environment'][key]==parent_control_on['train_environment'][key]
    else:
        # No actual same-host qualified ON currently exists. Keep this
        # alternative disabled rather than borrowing a different host's path
        # or pretending the cancelled extra OFF experiment is necessary.
        binding['cache_binding_status']='UNQUALIFIED_UNSTARTED_SAME_HOST_ON_NAMESPACE_NOT_DISPATCHABLE'
    provbytes=(json.dumps(prov,indent=2)+'\n').encode()
    assert sha(oldprov) in driver
    driver=driver.replace(sha(oldprov),sha(provbytes)).encode();ast.parse(driver)
    binding.update(schema='disabled-k2-qualified-base-control-v1',kind='control',base_sha256=source_sha,
        arm_id='K2-qualified-base-control',run_purpose='quality',prospective_record=proposal['prospective_record'],
        stage_authorized=False,launch_authorized=False,release_file_created=False,submission_authorized=False,
        full_proposal_present=True,full_proposal_path=proposal_path,full_proposal_sha256=sha(proposal_raw),
        driver_sha256=sha(driver),first_full_role_supersedes_unstarted_throughput_arm=True,
        actual_same_host_on_qualification_pending=True,actual_k2_speed_pass=False,
        source_adoption=False,no_automatic_next_job=True)
    binding['conditional_requirements']=[
        'Exactly one chosen first full alternative, with this prospective kind=control/base63925519 role frozen before actual launch; no duplicate throughput-arm record.',
        'Actual valid K2 native speed>=1.03 against its precommitted same-host window, native correctness/source/accounting and explicit quiescence accepted by Supervisor; no presumed PASS.',
        'trn22 uses the actual ON-parentcontrol namespace only after that run qualifies. trn21 remains disabled without a qualified same-host ON cache binding.',
        'Original lifetime lock, ps/neuron-ls/fuser/STOP/exact-source/argv/runtime and unchanged13.5GiB initial/10.5GiB runtime reserve; current/short artifacts retained.',
        'Full1800 plus fresh20M unchanged; actual role remains control regardless of BPB, no automatic adoption or submission.',
        'P2 must bind this actual retrieved same-host control row. A different host requires its own genuinely measured control.']
    files.update({'host_driver.py':driver,'package/provenance.json':provbytes,'EXPERIMENT_PROPOSAL.json':proposal_raw})
    files['NATIVE_BINDING.disabled.json']=(json.dumps(binding,indent=2)+'\n').encode()
    for n in ('package/train.py','package/launch-command.txt','lifetime_guard.py','idle_guard.py','dispatch_driver.py'):
        assert sha(files[n])==original['files'][n]['sha256']
    packet={'binding':binding,'files':{n:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for n,v in files.items()},
        'local_only':True,'host_calls':0,'native_started':False,
        'mechanical_derivation':{'parent_packet_sha256':sha(raw),'source_argv_and_guardian_unchanged':True,
            'driver_logic_changed':False,'changes':['prospective control role/provenance hash','trn22 cache owner new ON-parentcontrol label only','completed Supervisor proposal']}}
    out=HERE/(host+'-prepared.json');assert not out.exists()
    with out.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
    rows.append({'host':host,'label':binding['label'],'packet_sha256':sha(out.read_bytes()),'driver_sha256':sha(driver),
                 'kind':'control','base_sha256':source_sha,'cache_owner_label':binding['cache_owner_label'],
                 'state':'DISABLED_PENDING_ACTUAL_K2_QUALIFICATION_AND_SEPARATE_FULL_RELEASE'})
with (HERE/'PREPARED.json').open('x') as f:json.dump({'proposal_sha256':sha(proposal_raw),'one_selected_first_full_only':True,'packets':rows,'host_calls':0},f,indent=2);f.write('\n')
print(json.dumps({'status':'TWO_DISABLED_PROSPECTIVE_CONTROL_ALTERNATIVES','packets':rows}))
