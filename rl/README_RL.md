# Residual RL

This folder implements residual RL on top of the existing `StudentDecision`
controller. RL does not learn driving from scratch. Every control tick first
computes the base action, then applies a clipped residual:

```text
final_action = base_action + clipped_residual_action
```

The user-confirmed strongest baseline for this experiment is:

- raw time: `25.36s`
- penalty time: `3.00s`
- final time: `28.36s`

`base_policy=strongest` applies the baseline profile:

- `SCENARIO3_CRUISE_SPEED_LIMIT=0.80`
- `SCENARIO3_FINAL_SPEED_LIMIT=0.82`
- `COW_CLEARANCE_HOLD_SECONDS=3.00`
- `STOP_SIGN_HOLD_SECONDS=0.00`
- `CONE_PRE_X=1.950`

`base_policy=fast_safe` is the current speed-oriented safe profile for further
learning:

- `SCENARIO3_CRUISE_SPEED_LIMIT=1.40`
- `SCENARIO3_FINAL_SPEED_LIMIT=1.45`
- `COW_CLEARANCE_HOLD_SECONDS=2.20`
- `COW_CLEARANCE_MAX_X=0.70`
- `COW_ENTRY_TARGET_Y=4.520`
- `COW_CLEARANCE_TARGET_Y=4.520`
- `RL_ACTOR_RESIDUAL_ENABLED=0`

Hard pass requires deterministic evaluation to satisfy:

- `mean_final_time < 28.36`
- `mean_penalty_time <= 3.00`
- `invalid_episode_count == 0`

Latest local evaluation under the stricter current pedestrian/cow waiting
logic:

- `base_policy=strongest`, residual SAC `runs/residual_rl_penalty004_v2/best_model.zip`:
  `raw_time=26.56s`, `penalty_time=18.00s`, `collision_count=0`.
- `base_policy=fast_safe`, same residual model:
  `raw_time=21.95s`, `penalty_time=18.00s`, `collision_count=0`.

The current best direction is to train residual control from `fast_safe`, while
keeping cow residual disabled unless a separate safety gate proves no cow
collision. A short SAC continuation with pedestrian residual enabled did not
beat the older residual model, so it is archived but not deployed.

## Commands

Install local RL dependencies into the Quanser Python environment if needed:

```powershell
.\quanser\python.exe -m pip install -r rl\requirements_rl.txt
```

Baseline report:

```powershell
python rl/run_baselines.py --scenario 3 --out runs/residual_rl/baseline_report.json
```

Zero residual check:

```powershell
python rl/eval_residual_policy.py --scenario 3 --model zero --episodes 3 --out runs/residual_rl/zero_residual_eval.json --base_policy strongest
```

SAC training:

```powershell
python rl/train_residual_sac.py --scenario 3 --total_steps 20000 --save_dir runs/residual_rl --seed 0 --base_policy strongest --reward_minor_threshold 0.04
```

Fast-safe SAC continuation:

```powershell
python rl/train_residual_sac.py --scenario 3 --total_steps 20000 --save_dir runs/residual_rl_fast_safe --seed 0 --base_policy fast_safe --reward_minor_threshold 0.04 --init_model runs/residual_rl_penalty004_v2/best_model.zip
```

Deterministic evaluation:

```powershell
python rl/eval_residual_policy.py --scenario 3 --model runs/residual_rl/best_model.zip --episodes 5 --out runs/residual_rl/eval.json --base_policy strongest --auto_reenter_plane
```

Fast-safe deterministic evaluation:

```powershell
python rl/eval_residual_policy.py --scenario 3 --model runs/residual_rl_penalty004_v2/best_model.zip --episodes 1 --out runs/residual_rl_penalty004_v4_actor_micro/eval_fast_safe_policy.json --base_policy fast_safe --max_steps 5000 --auto_reenter_plane
```

## Safety/validity notes

- `QCarResidualEnv.step()` advances one control tick.
- The environment enables `core.single_tick_mode` so blocking waits and burst
  maneuvers are represented as cross-step state instead of being executed
  inside one `step()`.
- `PenaltySystem.py` thresholds are not changed by RL.
- `REWARD_MINOR_DEVIATION_THRESHOLD = 0.04` is used only for reward shaping.
- If QLabs or SAC dependencies fail, evaluation writes invalid episodes and
  `hard_pass=false`; it does not report fake success.
- If the vehicle state does not advance during warmup, reset fails with a
  message asking to exit and re-enter Plane. This prevents stuck QLabs scenes
  from being counted as valid evaluation.
- Training enables DPI-aware `Self-Driving -> Plane` re-entry after repeated
  reset failure. This does not restart QLAB and does not move or resize the
  QLAB window.
- `rl/qlab_reentry.py` exists only for that scene re-entry recovery path; it is
  not a QLAB restart script.
