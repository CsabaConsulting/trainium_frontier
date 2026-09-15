"""Exactness harness for plan §11.1's three schedule-level tricks (Recursive's recipe).

The three items -- Demon beta1 for the AdamW groups, the NorMuon beta2 ramp for the Muon groups,
and per-depth Muon LR/momentum scaling -- are all pure optimizer-schedule changes: no new tensors,
no new ops, no new parameters. This file checks the four claims that has to mean, all on the real
model at the locked config, CPU fp32, bitwise where bitwise is the standard:

 (1) ALL THREE FLAGS OFF == THE PRE-EDIT FILE, bitwise. Checked against the ACTUAL previous
     train.py (`git show <ref>:train.py`, imported as a second module) rather than against a
     hand-copied reference body, because this edit touched five places -- the beta2 literal in
     setup_optimizer, initial_betas, _muon_compile_scalars' staging tensor (4 -> 5 slots), the
     eager Muon momentum lerp, and the per-step schedule block -- and a hand copy of five places
     is exactly where a "reference" stops being independent. Run over both foreach settings and
     both lr representations.

 (2) DEMON BETA1 reaches only the AdamW groups. With it on, every AdamW-group parameter must move
     and every Muon-group and RMSProp-group parameter must be bitwise unchanged (the grads are
     supplied from outside, so the groups are genuinely independent here). Plus the schedule's own
     numbers: beta1 flat until the warmdown starts, then linear onto FINAL for a 0.8 group.

 (3) MUON BETA2 RAMP reaches only the Muon groups, and its value schedule is the stated ramp.

 (4) PER-DEPTH SCALING, three separate claims:
     (4a) plumbing: with NEUTRAL multipliers (mom bottom == top == ref, lr 1.0) the [K,1,1]
          operand path is bitwise identical to the 0-dim path it replaces. This is the one that
          catches a broadcasting or rounding mistake, because the answer is known exactly.
     (4b) a UNIFORM lr multiplier c equals scaling MATRIX_LR by c (to fp32 rounding, not bitwise:
          lr*ratio*c and (lr*c)*ratio round differently).
     (4c) a real ramp actually ramps: per-layer parameter deltas must be monotone in layer index
          in the direction the multipliers say, and the depth map must assign every block matrix
          its own block index.

Run: python research/scratch/sched_tricks_equiv.py [git-ref-or-path-of-the-pre-edit-train.py]

The argument defaults to the git ref "HEAD~1". A PATH is accepted too, because the Trn2 box's
checkout is not a clone of origin (no remote; files arrive by scp), so `git show` there would
resolve against an unrelated history.
"""
import copy
import importlib.util
import os
import subprocess
import sys

os.environ.setdefault("NEURON_COMPILE_CACHE_DIR", "/tmp/neuron_cache")
_ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD~1"
sys.argv = ["equiv"]  # keep train.py's argparse out of this

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import train as T

STEPS = 6
SEED = 1234


def load_preedit(ref: str):
    """Import the pre-edit train.py as a separate module so the OFF path can be compared against
    the real previous file instead of a hand-copied approximation.

    `ref` is a path if one exists at that name, otherwise a git ref resolved with `git show`.
    """
    path = "/tmp/train_preedit_sched.py"
    if os.path.exists(ref):
        with open(ref) as f:
            src = f.read()
    else:
        src = subprocess.run(["git", "show", f"{ref}:train.py"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("train_preedit_sched", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_preedit_sched"] = mod
    spec.loader.exec_module(mod)
    return mod


def build(mod):
    torch.manual_seed(SEED)
    return mod.GPT(mod.build_config())


def make_grads(model, step):
    g = torch.Generator().manual_seed(9000 + step)
    return {n: (torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.01).to(p.dtype)
            for n, p in model.named_parameters()}


def run(mod, model, grads_per_step, *, progress_of_step=None):
    """STEPS optimizer steps, driving the per-step schedule writes the training loop does.

    `progress_of_step(s)` supplies the wall-clock progress fraction; the default sweeps 0 -> 1 so
    every schedule under test is actually exercised over its whole range within STEPS steps.
    """
    if progress_of_step is None:
        def progress_of_step(s):
            return s / max(1, STEPS - 1)
    opt = model.setup_optimizer()
    named = dict(model.named_parameters())
    for s in range(STEPS):
        for name, p in named.items():
            p.grad = grads_per_step[s][name].clone().to(p.dtype)
        prog = progress_of_step(s)
        demon = mod.demon_beta1_scale(prog) if hasattr(mod, "demon_beta1_scale") else 1.0
        mb2 = mod.muon_beta2(prog) if hasattr(mod, "muon_beta2") else None
        for grp in opt.param_groups:
            grp["lr"] = grp["initial_lr"] * (1.0 - 0.3 * prog)
            if grp["kind"] == "muon":
                grp["momentum"] = mod.muon_momentum(s, prog)
                if mb2 is not None:
                    grp["beta2"] = mb2
            elif grp["kind"] == "adamw" and getattr(mod, "DEMON_BETA1", False):
                b1, b2 = grp["initial_betas"]
                grp["betas"] = (b1 * demon, b2)
        opt.step()
    return opt


def compare(label, m_ref, m_new, opt_ref, opt_new, *, expect_differ=(), tol=0.0):
    print(f"\n--- {label} ---")
    worst, worst_where, nparam, ndiff = 0.0, "", 0, 0
    differed = set()
    for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_new.named_parameters()):
        assert n1 == n2, (n1, n2)
        nparam += 1
        d = (p1.float() - p2.float()).abs().max().item()
        if not torch.equal(p1, p2):
            differed.add(n1)
            ndiff += 1
            if n1 not in expect_differ and d > tol:
                print(f"   UNEXPECTED MISMATCH param {n1}: max abs {d:.3e}")
        if n1 not in expect_differ and d > worst:
            worst, worst_where = d, f"param {n1}"
    keys = ("exp_avg", "exp_avg_sq", "momentum_buffer", "second_momentum_buffer")
    nstate = 0
    ref_state = list(opt_ref.state.values())
    new_state = list(opt_new.state.values())
    for sr, sn in zip(ref_state, new_state):
        for k in keys:
            if k in sr and k in sn:
                nstate += 1
                d = (sr[k].float() - sn[k].float()).abs().max().item()
                if d > worst and not expect_differ:
                    worst, worst_where = d, f"state[{k}]"
    print(f"   compared {nparam} params + {nstate} state tensors over {STEPS} steps; "
          f"{ndiff} params differed")
    missing = set(expect_differ) - differed
    if missing:
        print(f"   WARNING: expected these to differ and they did NOT: {sorted(missing)}")
    print(f"   max abs diff outside the expected set = {worst:.3e}  ({worst_where or 'exact'})")
    ok = worst <= tol and not missing
    print(f"   VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


def group_names(model):
    """{'adamw': set(names), 'rmsprop': ..., 'muon': ...} at the CURRENT flag settings."""
    opt = model.setup_optimizer()
    ids = {id(p): n for n, p in model.named_parameters()}
    out: dict[str, set[str]] = {}
    for g in opt.param_groups:
        out.setdefault(g["kind"], set()).update(ids[id(p)] for p in g["params"])
    return out, opt


def reset_flags():
    T.DEMON_BETA1 = False
    T.DEMON_BETA1_FINAL = 0.55
    T.DEMON_BETA1_REF = 0.8
    T.DEMON_BETA1_START_FRAC = -1.0
    T.MUON_BETA2 = 0.90
    T.MUON_BETA2_FINAL = 0.0
    T.MUON_DEPTH_LR_BOTTOM = 1.0
    T.MUON_DEPTH_LR_TOP = 1.0
    T.MUON_DEPTH_MOM_BOTTOM = 0.0
    T.MUON_DEPTH_MOM_TOP = 0.0
    T.MUON_DEPTH_MOM_REF = 0.95
    T.USE_FOREACH_OPTIM = True
    T.USE_TENSOR_LR_SCALARS = True


print(f"torch {torch.__version__}  |  locked config DEPTH={T.DEPTH} ASPECT_RATIO={T.ASPECT_RATIO} "
      f"USE_COMPILE_MUON={T.USE_COMPILE_MUON} USE_NKI_MUON={T.USE_NKI_MUON}")
all_ok = True
PRE = load_preedit(_ref)
print(f"pre-edit reference: git {_ref}:train.py  (MUON_BETA2 present={hasattr(PRE, 'MUON_BETA2')})")

# ---- (1) all three off == the pre-edit file ------------------------------------------------
for foreach in (True, False):
    for tensor_lr in (True, False):
        reset_flags()
        T.USE_FOREACH_OPTIM = PRE.USE_FOREACH_OPTIM = foreach
        T.USE_TENSOR_LR_SCALARS = PRE.USE_TENSOR_LR_SCALARS = tensor_lr
        m_ref = build(PRE)
        m_new = build(T)
        grads = [make_grads(m_ref, s) for s in range(STEPS)]
        opt_ref = run(PRE, m_ref, grads)
        opt_new = run(T, m_new, grads)
        all_ok &= compare(f"(1) all OFF vs pre-edit  foreach={foreach} tensor_lr={tensor_lr}",
                          m_ref, m_new, opt_ref, opt_new)

# ---- (2) Demon beta1: only the AdamW groups move ------------------------------------------
reset_flags()
m_off = build(T)
names, _ = group_names(m_off)
adamw_names = sorted(names.get("adamw", ()))
print(f"\nAdamW-group params ({len(adamw_names)}): {adamw_names}")
print(f"RMSProp-group params: {len(names.get('rmsprop', ()))}; "
      f"Muon-group params: {len(names.get('muon', ()))}")
m_on = copy.deepcopy(m_off)
grads = [make_grads(m_off, s) for s in range(STEPS)]
opt_off = run(T, m_off, grads)
T.DEMON_BETA1 = True
opt_on = run(T, m_on, grads)
all_ok &= compare("(2) demon beta1 ON vs OFF -- only AdamW-group params may move",
                  m_off, m_on, opt_off, opt_on, expect_differ=tuple(adamw_names))

print("\n--- (2b) the beta1 schedule's own numbers (start = 1 - WARMDOWN_RATIO = "
      f"{1 - T.WARMDOWN_RATIO:.2f}) ---")
sched_ok = True
for prog in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
    s = T.demon_beta1_scale(prog)
    print(f"   progress {prog:4.2f}  scale {s:.6f}  beta1(0.8 group) {0.8 * s:.6f}  "
          f"beta1(0.9 group) {0.9 * s:.6f}")
sched_ok &= T.demon_beta1_scale(0.0) == 1.0 and T.demon_beta1_scale(0.25) == 1.0
sched_ok &= abs(0.8 * T.demon_beta1_scale(1.0) - T.DEMON_BETA1_FINAL) < 1e-12
mid = 0.8 * T.demon_beta1_scale(0.625)  # halfway through [0.25, 1]
sched_ok &= abs(mid - 0.5 * (0.8 + T.DEMON_BETA1_FINAL)) < 1e-12
print(f"   flat before the warmdown, exactly {T.DEMON_BETA1_FINAL} at progress 1, linear in "
      f"between: {'PASS' if sched_ok else 'FAIL'}")
all_ok &= sched_ok

# ---- (3) Muon beta2 ramp: only the Muon groups move ---------------------------------------
reset_flags()
m_off = build(T)
names, _ = group_names(m_off)
muon_names = sorted(names.get("muon", ()))
m_on = copy.deepcopy(m_off)
grads = [make_grads(m_off, s) for s in range(STEPS)]
opt_off = run(T, m_off, grads)
T.MUON_BETA2_FINAL = 0.97
opt_on = run(T, m_on, grads)
all_ok &= compare("(3) muon beta2 ramp ON vs OFF -- only Muon-group params may move",
                  m_off, m_on, opt_off, opt_on, expect_differ=tuple(muon_names))
print("\n--- (3b) the beta2 ramp's own numbers ---")
b2_ok = T.muon_beta2(0.0) == T.MUON_BETA2 and abs(T.muon_beta2(1.0) - 0.97) < 1e-12
for prog in (0.0, 0.25, 0.5, 1.0):
    print(f"   progress {prog:4.2f}  beta2 {T.muon_beta2(prog):.6f}")
T.MUON_BETA2_FINAL = 0.0
b2_ok &= T.muon_beta2(0.5) == T.MUON_BETA2
print(f"   endpoints exact and OFF is flat at MUON_BETA2: {'PASS' if b2_ok else 'FAIL'}")
all_ok &= b2_ok

# ---- (4a) per-depth plumbing with NEUTRAL multipliers == the 0-dim path, bitwise -----------
reset_flags()
m_off = build(T)
m_neutral = copy.deepcopy(m_off)
grads = [make_grads(m_off, s) for s in range(STEPS)]
opt_off = run(T, m_off, grads)
T.MUON_DEPTH_MOM_BOTTOM = T.MUON_DEPTH_MOM_TOP = T.MUON_DEPTH_MOM_REF  # multiplier == 1.0
opt_neutral = run(T, m_neutral, grads)
all_ok &= compare("(4a) per-depth ON with neutral multipliers vs OFF -- must be bitwise equal",
                  m_off, m_neutral, opt_off, opt_neutral)

# ---- (4b) a uniform lr multiplier == scaling MATRIX_LR ------------------------------------
# Run in fp32 COMPUTE_DTYPE. The claim being tested is "the [K,1,1] lr operand carries the same
# number the 0-dim one does", whose only slack is fp32 reassociation (lr*ratio*c vs (lr*c)*ratio) --
# ~1e-7 relative. Under the default bf16 compute dtype the Newton-Schulz iterate and hence the whole
# update carry an 8-bit mantissa, which is ~1e-3 relative and would swamp the thing under test: the
# check would then pass at a tolerance loose enough to hide a real per-parameter-operand bug.
reset_flags()
C = 1.3
saved_dtype = T.COMPUTE_DTYPE
T.COMPUTE_DTYPE = torch.float32
m_mult = build(T)
grads = [make_grads(m_mult, s) for s in range(STEPS)]
T.MUON_DEPTH_LR_BOTTOM = T.MUON_DEPTH_LR_TOP = C
opt_mult = run(T, m_mult, grads)
reset_flags()
T.COMPUTE_DTYPE = torch.float32
saved_lr = T.MATRIX_LR
T.MATRIX_LR = saved_lr * C
m_lr = build(T)
opt_lr = run(T, m_lr, grads)
T.MATRIX_LR = saved_lr
all_ok &= compare(f"(4b) uniform depth lr x{C} vs MATRIX_LR x{C} -- equal to fp32 rounding",
                  m_lr, m_mult, opt_lr, opt_mult, tol=2e-7)
T.COMPUTE_DTYPE = saved_dtype

# ---- (4c) a real ramp ramps, and the depth map is right -----------------------------------
print("\n--- (4c) the depth map and the ramp's direction ---")
reset_flags()
m_base = build(T)
_, opt_probe = group_names(m_base)
ids = {id(p): n for n, p in m_base.named_parameters()}
T.MUON_DEPTH_LR_BOTTOM, T.MUON_DEPTH_LR_TOP = 1.15, 0.85
T.MUON_DEPTH_MOM_BOTTOM, T.MUON_DEPTH_MOM_TOP = 0.90, 0.97
m_depth = build(T)
opt_depth = m_depth.setup_optimizer()
map_ok, ramp_ok = True, True
for g in opt_depth.param_groups:
    if g["kind"] != "muon":
        continue
    lr_mult, mom_mult = g["depth_mults"]
    rows = []
    for p, lm, mm in zip(g["params"], lr_mult, mom_mult):
        n = ids[id(p)]
        want = None
        if n.startswith("transformer.h."):
            want = int(n.split(".")[2])
        rows.append((n, want, lm, mm))
        if want is None:
            if lm != 1.0 or mm != 1.0:
                print(f"   BAD: non-block param {n} got multipliers {lm}/{mm}, expected 1.0/1.0")
                map_ok = False
        else:
            f = want / max(1, T.DEPTH - 1)
            exp_lm = 1.15 + f * (0.85 - 1.15)
            exp_mm = (1.0 - (0.90 + f * (0.97 - 0.90))) / (1.0 - 0.95)
            if abs(lm - exp_lm) > 1e-12 or abs(mm - exp_mm) > 1e-12:
                print(f"   BAD: {n} depth {want} got {lm:.6f}/{mm:.6f}, "
                      f"expected {exp_lm:.6f}/{exp_mm:.6f}")
                map_ok = False
    depths = [r[1] for r in rows]
    lms = [r[2] for r in rows]
    if len(rows) > 1 and all(d is not None for d in depths):
        if not all(a >= b for a, b in zip(lms, lms[1:])) or depths != sorted(depths):
            print(f"   BAD: group {tuple(g['params'][0].shape)} multipliers not monotone by depth: "
                  f"{list(zip(depths, lms))}")
            ramp_ok = False
    print(f"   group {str(tuple(g['params'][0].shape)):>14}  K={len(rows):2d}  "
          f"depths={depths}  lr_mult={[round(x, 4) for x in lms]}")
# and the end-to-end effect: layer 0's matrices must move MORE than the last layer's
grads = [make_grads(m_base, s) for s in range(STEPS)]
m_a, m_b = copy.deepcopy(m_base), copy.deepcopy(m_base)
reset_flags()
run(T, m_a, grads)
T.MUON_DEPTH_LR_BOTTOM, T.MUON_DEPTH_LR_TOP = 1.15, 0.85
T.MUON_DEPTH_MOM_BOTTOM, T.MUON_DEPTH_MOM_TOP = 0.90, 0.97
run(T, m_b, grads)
base = dict(m_base.named_parameters())
pa, pb = dict(m_a.named_parameters()), dict(m_b.named_parameters())
probe_name = "attn.c_proj.weight"
print(f"   |delta| of transformer.h.<i>.{probe_name} relative to the OFF arm:")
rel = []
for i in range(T.DEPTH):
    n = f"transformer.h.{i}.{probe_name}"
    if n not in pa:
        continue
    da = (pa[n].float() - base[n].float()).abs().mean().item()
    db = (pb[n].float() - base[n].float()).abs().mean().item()
    rel.append(db / max(1e-30, da))
    print(f"      layer {i}: OFF {da:.6e}  DEPTH {db:.6e}  ratio {rel[-1]:.4f}")
if rel:
    mono = all(a > b for a, b in zip(rel, rel[1:]))
    print(f"   ratio strictly decreasing with depth (1.15 -> 0.85 says it must be): "
          f"{'PASS' if mono else 'FAIL'}")
    ramp_ok &= mono
print(f"   depth map exact: {'PASS' if map_ok else 'FAIL'}; ramp direction: "
      f"{'PASS' if ramp_ok else 'FAIL'}")
all_ok &= map_ok and ramp_ok

reset_flags()
print(f"\n================ OVERALL: {'ALL CHECKS PASS' if all_ok else 'FAILURE'} ================")
sys.exit(0 if all_ok else 1)
