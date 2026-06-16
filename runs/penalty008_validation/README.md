# 0.08 penalty validation traces

These traces were generated with:

```powershell
$env:PENALTY_MINOR_DEVIATION_THRESHOLD = "0.08"
.\quanser\python.exe rl\trace_residual_policy.py --base_policy fast_safe --model zero --auto_reenter_plane --out runs\<trace>.csv
```

Validated results:

| Trace | Raw time | Penalty | Final time | Penalty rows |
| --- | ---: | ---: | ---: | --- |
| `trace_fast_safe_penalty008_cowexit_soft_20260615_180719.csv` | 28.56s | 3.00s | 31.56s | cone area only |
| `trace_fast_safe_penalty008_cowexit_soft_stability2_20260615_180851.csv` | 28.22s | 3.00s | 31.22s | cone area only |
| `trace_fast_safe_penalty008_cowexit_soft_stability3_20260615_181222.csv` | 28.06s | 3.00s | 31.06s | cone area only |
| `trace_current_penalty008_20260616_182020.csv` | 28.35s | 3.00s | 31.35s | cone area only |

Large local runtimes, models, QLAB screenshots, and generated logs are intentionally not committed.
