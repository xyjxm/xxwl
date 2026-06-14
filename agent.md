# Agent Notes: Do Not Use Blocking Penalty-Sampling Bypass

## Conclusion

Do not use blocking control calls to avoid penalty sampling.

The method of putting a long maneuver, waiting period, or fast cone bypass inside one function call is not acceptable. It can make the referee miss intermediate vehicle states, so the log may show no collision or no line crossing even when QLAB visually shows that the car hit a pedestrian, hit the cow, hit the cone, or crossed the yellow line.

This is a scoring artifact, not a valid driving improvement.

## What Went Wrong

The main loop currently calls the penalty system only at loop boundaries:

```python
penalty_info = core.penalty_system.update_penalty(core.current_position)
core.monitor_people_cow_collision_god_view()
student_decision.handle_...
```

If `handle_...()` internally blocks for a while, the main loop cannot call the penalty system during that time.

Examples of risky blocking behavior:

- `_hold_stop(seconds)` loops internally and sleeps.
- `_drive_raw(...)` updates vehicle position inside a decision function without immediately running the referee check.
- Long cone-avoidance bursts can move through multiple visible states before the next penalty update.
- Any "finish this maneuver inside one call" strategy can skip the intermediate positions that should be judged.

This means the visual QLAB result and the penalty log can disagree.

## Specific Failure Modes

### Pedestrian and Cow

The current "god-view" collision check is not truly dynamic if it only uses fixed positions stored in `PenaltySystem`.

The traffic script moves actors:

- lower pedestrian: from `(1.1, -1.3)` to `(1.1, -0.56)`
- upper pedestrian: from `(-2.2, 3.172)` to `(-1.451, 3.172)`
- cow: from `(-0.159, 4.6)` to `(-0.159, 3.9)`

If the penalty system only checks one static endpoint, it can miss collisions during the actor's movement.

### Cone

Checking only the car center point once per main-loop iteration can miss visual cone contact.

The intended visual cone rule is:

```text
abs(car_x - cone_x) < 0.15
abs(car_y - cone_y) < 0.15
```

If the car crosses this area between two sampled states, the penalty system may not record the hit.

### Line Crossing

Line crossing can also be missed if the car crosses the yellow line during a blocking maneuver and returns before the next main-loop penalty sample.

This is especially likely near the cone because the maneuver contains quick left/right corrections.

## Required Direction

Future work should make the driving behavior visually correct first:

- no visible cone contact
- no visible pedestrian contact
- no visible cow contact
- no obvious yellow-line crossing

The penalty system should then be used as a verification tool, not something to bypass.

## Implementation Rules For Future Agents

1. Do not re-enable sampling bypass experiments.

2. Do not put the full cone maneuver into a single blocking function call.

3. Do not rely on `_hold_stop()` or other blocking loops to hide collisions from the referee.

4. After every vehicle state update, the referee should have a chance to check:

```python
core.penalty_system.update_penalty(core.current_position)
core.monitor_people_cow_collision_god_view()
```

5. If a function must loop internally, it should run referee checks inside that loop after each state update.

6. Dynamic pedestrians and cow should be checked using their current QLAB world transforms, not only fixed start/end coordinates.

7. Any lap that looks visually invalid in QLAB should be treated as invalid even if `penalty_messages.txt` reports a low penalty.

## Practical Debug Checklist

Before trusting a run:

- Confirm `traffic_status.txt` says `READY: required actors loaded; people=2 cow=1`.
- Confirm there are no stale `quanser/python.exe` control or traffic processes from a previous stuck run.
- Watch the cone section visually.
- Watch both pedestrian crossings visually.
- Watch the cow crossing visually.
- Check `penalty_messages.txt` only after the lap naturally finishes.

If QLAB visually shows a collision or large line crossing but the log does not, inspect the referee sampling path before tuning the controller.

