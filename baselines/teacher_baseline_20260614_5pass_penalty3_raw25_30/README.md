# Teacher Baseline 2026-06-14

Scenario: 3

Traffic:
- `TRAFFIC_ACTOR_MODE=triggered`
- `TRAFFIC_LIGHT_MODE=fixed_green`
- required actors loaded every run: `people=2 cow=1`

Pass criteria:
- 5 consecutive valid runs
- raw lap time between 25s and 30s
- penalty time <= 3s

Validation archive:

`run_archives/20260614_011907_fixed_green_5run_validation_reenter_fullwait`

Results:

| Run | Raw time (s) | Penalty (s) | Pass |
| --- | ---: | ---: | --- |
| 1 | 25.2341964244843 | 3 | true |
| 2 | 25.285905122757 | 3 | true |
| 3 | 25.35822057724 | 3 | true |
| 4 | 25.2149469852448 | 3 | true |
| 5 | 25.2907538414001 | 3 | true |

Baseline raw time to beat:

`25.2149469852448s`

Notes:
- QLAB was not restarted.
- Plane was re-entered before each validation run to refresh the scene.
- The line penalty logic keeps continuous sampling and a 1.5s continuous deviation interval.
- A single continuous cone-bypass line deviation is not double-counted solely because it crosses the cone before/after section boundary.
