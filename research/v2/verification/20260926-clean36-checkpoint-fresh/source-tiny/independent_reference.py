import subprocess
import torch

torch.set_num_threads(1)
source = subprocess.check_output(['git','show','0930f2ae:research/v2/proposals/20260926-clock36-checkpoint-mask-r3/compare_checkpoints.py'], cwd='/home/poojith-devan/Desktop/Trainium', text=True)
ns = {'__name__': 'reviewed_baseline'}
exec(compile(source, 'immutable_baseline', 'exec'), ns)
passed=[]
def reject(label, fn):
    try:
        fn()
    except RuntimeError:
        passed.append(label)
    else:
        raise AssertionError('accepted '+label)

# Deliberately use scalar loops rather than the helper's broadcast formula.
for length in (4,7,65):
    for window in (1,3,length):
        mask=torch.full((1,1,length,length),float('-inf'))
        for row in range(length):
            for col in range(max(0,row-window+1),row+1):
                mask[0,0,row,col]=0
        ns['mask_record'](mask,length,window,'independent')
        for label, coordinate, value in [('wrong_zero',(0,1),0),('nan',(0,0),float('nan')),('posinf',(0,0),float('inf')),('negative_zero',(0,0),-0.0)]:
            bad=mask.clone(); bad[0,0,coordinate[0],coordinate[1]]=value
            reject(label,lambda: ns['mask_record'](bad,length,window,'bad'))

class Fixture(torch.nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode=mode
        self.weight=torch.nn.Parameter(torch.ones(1))
        self.register_buffer('scratch',torch.zeros(1),persistent=False)
        self.calls=0
    def forward(self,x):
        self.calls+=1
        if self.mode=='weight': self.weight.add_(1)
        if self.mode=='buffer': self.scratch.add_(1)
        if self.mode=='input': x[0,0]+=1
        y=torch.nn.functional.one_hot(x,16).float()
        if self.mode=='noncausal': y=y+x[:,-1,None,None].float()
        if self.mode=='shape': y=y[:,:,:-1]
        if self.mode=='alias':
            if not hasattr(self,'out'): self.out=y.clone()
            self.out.copy_(y+self.calls)
            y=self.out
        return y

x=torch.tensor([[1,2,3,4,5,6,7,8]])
ns['verify_trained_model'](Fixture('valid').eval(),x.clone(),16)
for mode in ('weight','buffer','input','noncausal','shape','alias'):
    reject(mode,lambda: ns['verify_trained_model'](Fixture(mode).eval(),x.clone(),16))
print({'scope':'independent tiny tests against baseline context only; candidate pending','negative_checks':len(passed),'labels':sorted(set(passed)),'torch':torch.__version__})
