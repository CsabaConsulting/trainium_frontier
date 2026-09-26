import ast,hashlib,json
from pathlib import Path
p=Path(__file__).parent
trees={n:ast.parse((p/(n+'.py')).read_text()) for n in ('vehicle','candidate','research')}
def fn(which,name):return next(n for n in trees[which].body if isinstance(n,ast.FunctionDef) and n.name==name)
def canonical(n):return ast.dump(n,include_attributes=False)
assert canonical(fn('vehicle','load_for_eval'))==canonical(fn('candidate','load_for_eval'))
assert canonical(fn('research','startup_adjusted_schedule_elapsed'))==canonical(fn('candidate','startup_adjusted_schedule_elapsed'))
assert canonical(fn('vehicle','save_checkpoint'))==canonical(fn('candidate','save_checkpoint'))
main=fn('candidate','main')
loop=next(n for n in main.body if isinstance(n,ast.While))
boundary=next(n for n in loop.body if isinstance(n,ast.If) and 'TRAIN_STARTUP_STEPS_EXCLUDED' in ast.unparse(n.test))
class Clock:
    def __init__(self):self.calls=0
    def monotonic(self):self.calls+=1;return 1050.
for step in range(7):
    time=Clock(); ns=dict(time=time,step=step,TRAIN_STARTUP_STEPS_EXCLUDED=5,_PROCESS_STARTED=100.,TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900.,STARTUP_LAUNCH_MARGIN_SECONDS=30.,budget_started=200.,recent_step_times=[11.,12.,13.])
    exec(compile(ast.Module([boundary],type_ignores=[]),'actual-boundary','exec'),ns)
    assert time.calls==(1 if step==4 else 0)
    assert ns['recent_step_times']==([] if step==4 else [11.,12.,13.])
    assert ns['budget_started']==(970. if step==4 else 200.)
assert not any(isinstance(n,ast.Call) and 'monotonic' in ast.unparse(n.func) for n in ast.walk(fn('candidate','startup_adjusted_schedule_elapsed')))
assert hashlib.sha256((p.parent/'launch-command.txt').read_bytes()).hexdigest()=='b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd'
print(json.dumps({'unchanged_ordinary_loader':True,'unchanged_save':True,'identical_research_schedule_helper':True,'actual_boundary_cases':7,'boundary_clock_calls':'one at fifth boundary only','launch_hash':'PASS'}))
