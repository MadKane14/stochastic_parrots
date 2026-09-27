# p2's queued T14_full India dense CV — run by p4 on p4's Kaggle: FAILED (out of memory) — Sun 00:00

Status: failed, no result. Not rerun (reason below).
Kernel `anishtoshniwal006/ber-p4-p2full` (2× T4, 4 cores, 31 GB), p2's commit `a12b762` unchanged,
`python exploration/p2/dense_cv_group.py -c code/business_entity_resolution/configs/exp/p2/T14_full.yaml --group India`.
Logs: `exploration/p4/p2_T14full/logs/` (`dense_cv_India_T14_full.log`, `summary.txt`).

## What happened (from the log)
| stage | time | note |
|---|---|---|
| normalize | 185 s | |
| block (p1_compound + p3_chargram) | 1,568 s | 30,459,509 pairs (883,188 India S1 × 4.13M right) |
| features p1_match | 663 s | peak 24.6 GB |
| features p1_extra | 520 s | peak 24.6 GB |
| features p4_e5 | 2,005 s | embedded 3,678,909 names (9,527/s) + 4,986,146 name+addr (4,379/s) on a T4; **peak 31.1 GB** |
| (next step) | – | **killed, exit 137**, after 5,236 s total |

Cause: `p4_e5` keeps every vector of a field in RAM (as found in p4 T12). On top of p1_match/p1_extra for 30.5M
pairs, that exceeds Kaggle's 31 GB. The memory-mapped `p1_e5` (main) is the fix.

## Why not rerun
- Rerunning needs `p1_e5` instead of `p4_e5`, i.e. a change to p2's pre-registered T14_full config (p2's call).
- Round 5 covers the same question with the current stack: T18 (p3 blocking on the India group), T19 (p4_tcanon) and
  **T20 (fine-tuned vs pretrained E5 on T13_sub4 with `p1_e5`), which p4 is running for p2 now**
  (`reports/p4/T20_running_for_p2.md`). T20's baseline run is the pretrained full stack on the India group.
- GPU budget: the account is near its weekly limit after T19 + T20.
If p2 still wants T14_full: copy `configs/exp/p2/T14_full.yaml` with `p4_e5` → `p1_e5 {fields: [name, full]}` and
p4 can run it once a GPU slot is free (~1.5 h).
