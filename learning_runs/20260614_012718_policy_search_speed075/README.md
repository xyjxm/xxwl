# Policy Search Candidate: speed075

Method:
- Lightweight policy search over scenario 3 cruise speed.
- Baseline to beat: `25.2149469852448s` raw time with penalty <= 3s.

Candidate parameters:
- `SCENARIO3_CRUISE_SPEED_LIMIT=0.75`
- `SCENARIO3_FINAL_SPEED_LIMIT=0.75`
- `CONE_PRE_X=1.950`

Result:
- Raw time: `23.963897943496704s`
- Penalty: `3.0s`
- Required actors loaded: `people=2 cow=1`
- Cone collisions: `0`
- Pedestrian collisions: `0`
- Cow collisions: `0`
- Red-light violations: `0`

This candidate beats the saved baseline by `1.251049041748096s` raw time while keeping penalty <= 3s.
