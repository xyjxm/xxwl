# Learned Default Policy Verification

This run verifies the current default code after applying the learned `speed075` policy.

Conditions:
- `TRAFFIC_ACTOR_MODE=triggered`
- `TRAFFIC_LIGHT_MODE=fixed_green`
- no `SCENARIO3_CRUISE_SPEED_LIMIT` override
- no `SCENARIO3_FINAL_SPEED_LIMIT` override
- required actors loaded: `people=2 cow=1`

Result:
- Raw time: `23.964003324508667s`
- Penalty: `3.0s`
- Final time: `26.964003324508667s`

Comparison:
- Saved baseline best raw time: `25.2149469852448s`
- Improvement: `1.250943660736133s`

Penalty breakdown:
- minor line deviation: `1`
- major line deviation: `0`
- cone collisions: `0`
- continuous line penalty: `0.0`
- pedestrian collisions: `0`
- cow collisions: `0`
- red-light violations: `0`
