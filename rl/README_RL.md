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

- `SCENARIO3_CRUISE_SPEED_LIMIT=0.68`
- `SCENARIO3_FINAL_SPEED_LIMIT=0.68`
- `CONE_PRE_X=1.950`

Hard pass requires deterministic evaluation to satisfy:

- `mean_final_time < 28.36`
- `mean_penalty_time <= 3.00`
- `invalid_episode_count == 0`

Latest local evaluation:

- zero residual, 3 episodes: `mean_final_time=28.0696s`,
  `mean_penalty_time=3.00s`, `invalid_episode_count=0`
- residual SAC best model, 5 episodes: `mean_final_time=27.9865s`,
  `mean_penalty_time=3.00s`, `invalid_episode_count=0`, `hard_pass=true`

The deployed residual range is intentionally conservative. Residual control is
disabled in the cone area and pedestrian/cow area, and only a small speed boost
plus tiny steering bias is allowed on the final straight/finish approach. This
keeps the learned policy from changing the visually sensitive obstacle
avoidance behavior.

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

Deterministic evaluation:

```powershell
python rl/eval_residual_policy.py --scenario 3 --model runs/residual_rl/best_model.zip --episodes 5 --out runs/residual_rl/eval.json --base_policy strongest --auto_reenter_plane
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
