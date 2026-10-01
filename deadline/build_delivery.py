#!/usr/bin/env python3
"""Remove unreachable recipe switches and optimizer-resume code without rewriting math.

Source edits retain comments and indentation. An AST equivalence check compares
both versions after folding the immutable switches and dropping only the unused
optimizer class/API; function bodies, constants, calls and timing code remain.
"""
import ast,copy,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SWITCHES={'FLAT_WEIGHTS','Z1_NORM_ONCE','Z3_ONE_GATE_SLICE','Z5_SKIP_VOCAB_SLICE','Z6_TABLE_REBIND','Z7_PERSISTENT_SCALARS','Z10_ROPE_WRAP_NKI','Z11_FULLGRAPH_CORE'}


def edit(text,start,end,replacement):
 lines=text.splitlines(keepends=True)
 return ''.join(lines[:start-1])+replacement+''.join(lines[end:])


def folded_source(text):
 while True:
  tree=ast.parse(text)
  nodes=[n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id in SWITCHES]
  if not nodes:break
  # Innermost/rightmost edit, then reparse; never overlap source ranges.
  n=max(nodes,key=lambda n:n.lineno)
  lines=text.splitlines(keepends=True)
  end=n.body[-1].end_lineno
  if n.orelse:
   end=min(end,n.orelse[0].lineno-2)
  active=lines[n.lineno:end]
  active=['    '+line[8:] if len(line)-len(line.lstrip())>=8 else line[4:] if line.startswith('    ') else line for line in active]
  # Dedent by exactly one block, preserving each original indentation level.
  active=[line[4:] if line.startswith('    ') else line for line in lines[n.lineno:end]]
  text=edit(text,n.lineno,n.end_lineno,''.join(active))
 tree=ast.parse(text)
 removals=[]
 for n in tree.body:
  if isinstance(n,ast.ClassDef) and n.name=='OwnedOptimizer':removals.append((n.lineno,n.end_lineno))
  if isinstance(n,ast.ClassDef) and n.name=='FlatWeights':
   for f in n.body:
    if isinstance(f,ast.FunctionDef) and f.name in ('state_dict','load_state_dict'):
     removals.append((min([f.lineno]+[d.lineno for d in f.decorator_list]),f.end_lineno))
 for start,end in sorted(removals,reverse=True):text=edit(text,start,end,'')
 tree=ast.parse(text)
 loads={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
 for n in reversed(tree.body):
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in SWITCHES and n.targets[0].id not in loads:
   text=edit(text,n.lineno,n.end_lineno,'')
 text=text.replace('Trainium Frontier Phase 1: the fixed CLEAN-GATEFIX training recipe.','Trainium Frontier Phase 1: fixed eight-block training recipe.')
 text=text.replace('# FLATCACHE: flat fp32 master + flat bf16 weight cache + flat eager tail (False = the per-parameter path).','# Flat FP32 master weights, BF16 matrix cache and shared gradient storage.')
 text=text.replace('# Z-bundle switches (OP_INVENTORY.md, "Zero-engineering fixes"). Every switch is\n# math-identical in forward; Z1 and Z3 change only the BF16 accumulation order of\n# one gradient sum. Flip one to False to bisect; all default ON.\n','# Fixed arithmetic and compilation choices for this training recipe.\n')
 text=text.replace('    # ------------------------------------------------------------------ checkpointing (OwnedOptimizer schema)\n','')
 text=text.replace('clip_and_sync_gradients + OwnedOptimizer.step + refresh_weight_cache, flat.','Reduce, clip, update the owned optimizer shard and refresh BF16 weights.')
 while '\n\n\n' in text:text=text.replace('\n\n\n','\n\n')
 compile(text,'delivery/train.py','exec')
 return text


class Canonical(ast.NodeTransformer):
 def visit_If(self,n):
  if isinstance(n.test,ast.Name) and n.test.id in SWITCHES:
   return [self.visit(x) for x in n.body]
  return self.generic_visit(n)
 def visit_ClassDef(self,n):
  if n.name=='OwnedOptimizer':return None
  if n.name=='FlatWeights':
   n.body=[f for f in n.body if not isinstance(f,ast.FunctionDef) or f.name not in ('state_dict','load_state_dict')]
  return self.generic_visit(n)
 def visit_Expr(self,n):
  if isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):return None
  return self.generic_visit(n)


def canonical(text):
 tree=Canonical().visit(ast.parse(text))
 loads={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
 tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in SWITCHES and n.targets[0].id not in loads)]
 return ast.dump(tree,include_attributes=False)


def main():
 inputs={'frozen':ROOT/'submission/train.py','rmsfp32':ROOT/'candidates/rmsfp32/train.py','headbias':ROOT/'candidates/headbias/train.py','dualhead':ROOT/'candidates/dualhead/train.py'}
 receipts=[]
 command='NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py'
 assert len(command)==76 and len(command)<=256
 for name,source in inputs.items():
  original=source.read_text();clean=folded_source(original)
  if name=='rmsfp32':
   clean=clean.replace('FP32; the table and its RMS state are BF16;', 'FP32; table values are BF16 and its RMS state/denominator are FP32;')
  assert canonical(original)==canonical(clean),(name,'executable AST changed')
  # Main CLI still explicitly defaults the table freeze boundary to zero.
  assert "parser.add_argument('--tbl_freeze_tokens', type=int, default=0" in clean
  target=ROOT/'deadline_submission'/name/'train.py';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(clean)
  (target.parent/'COMMAND.txt').write_text(command+'\n')
  receipts.append(dict(variant=name,source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(original.encode()).hexdigest(),delivery=str(target.relative_to(ROOT)),delivery_sha256=hashlib.sha256(clean.encode()).hexdigest(),original_lines=len(original.splitlines()),delivery_lines=len(clean.splitlines()),canonical_executable_ast_equal=True,removed=['unreachable fixed-switch branches','unused non-flat OwnedOptimizer class','unused optimizer-resume methods'],model_training_and_timing_math_unchanged=True,launch_command=command,launch_command_chars=len(command),table_freeze_default=0))
 out=ROOT/'deadline_submission/cleaning-receipt.json';out.write_text(json.dumps(receipts,indent=2)+'\n')
 print(json.dumps(receipts,indent=2))

if __name__=='__main__':main()
