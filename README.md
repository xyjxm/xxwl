# residual_rl_line_following_snapshot_20260614_202559

This snapshot saves the current local experiment version after adding line-following tuning hooks.

## Saved State

- Project: `qimodazuoye_teacher/qimodazuoye`
- Saved at: `2026-06-14 20:25:59`
- Current strict penalty settings are preserved in `PenaltySystem.py`:
  - minor line threshold: `0.04`
  - continuous line interval: `1.5s`
  - cone visual/collision half side: `0.15`
- Known best verified run for this code line is still:
  - raw time: about `27.62s`
  - penalty: `9.0s`
  - final time: about `36.62s`
  - collision count: `0`
  - report: `runs/residual_rl_penalty004_v4_actor_micro/eval_keep_upper_window_only.json`

## Important Notes

- This is not yet the final `3s`-penalty target.
- The default behavior is intended to stay close to the current stable `9s` version.
- Large local runtimes/dependencies are intentionally not included:
  - `python/`, `quanser/`, `rtmodels/`
  - `qlabspretrained.pt`
  - trained model archives such as `best_model.zip`
  - full QLAB installation files
  These should remain available in the local lab environment when running the full simulator/camera stack.
- New tuning hooks are saved but default off or default-neutral:
  - `SCENARIO3_ARCLENGTH_TARGET=1`: optional continuous arclength lookahead target.
  - `RL_LINE_DAMPING_ENABLED=1`: optional residual steering damping near line limits.
  - `LOWER_CORNER_TARGET_Y`, `LOWER_CORNER_FORWARD_X`, `LOWER_CORNER_SPEED_CAP`: lower-left corner tuning.
  - `COW_ENTRY_TARGET_Y`, `COW_ENTRY_X_OFFSET`, `COW_CLEARANCE_TARGET_Y`, `COW_CLEARANCE_X_OFFSET`: cow-area line tuning.
  - `RL_STRICT_LANE_LOWER_LEFT_MAX`, `RL_STRICT_LANE_LOWER_LEFT_GAIN`: local strict-lane shield tuning.
- Recent experiments showed that broad/global line-following changes easily make cone/cow behavior worse. Future work should prefer local or learned residual fixes.

## Reproduce Current Stable Eval

From the project root:

The command below expects the local lab machine to already have the omitted
model file at `runs/residual_rl_penalty004_v2/best_model.zip`.

```powershell
$env:PYTHONIOENCODING='utf-8'
Remove-Item Env:\SCENARIO3_ARCLENGTH_TARGET -ErrorAction SilentlyContinue
Remove-Item Env:\RL_LINE_DAMPING_ENABLED -ErrorAction SilentlyContinue
Remove-Item Env:\RL_ACTOR_RESIDUAL_ENABLED -ErrorAction SilentlyContinue
.\quanser\python.exe rl\eval_residual_policy.py --scenario 3 --model runs\residual_rl_penalty004_v2\best_model.zip --episodes 1 --out runs\residual_rl_penalty004_v4_actor_micro\eval_snapshot_rerun.json --base_policy strongest --max_steps 5000 --auto_reenter_plane
```

## Included Files

- Core controller and referee files.
- `rl/*.py` residual RL environment/training/evaluation files.
- Residual RL config and key reports. Model archives are intentionally omitted from GitHub.
- Key reports/traces from the recent tuning pass.
