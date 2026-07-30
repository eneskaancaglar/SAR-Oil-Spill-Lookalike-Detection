# Known Limitations

1. The v0.8 independent negative holdout produced **2/40** false-alarm scenes.
2. No untouched positive `oc/ow` holdout remained, so independent positive recall could not be measured.
3. The water gate intentionally blocks uncertain scenes; this improves safety but reduces coverage.
4. Calibration performance must not be reported as independent test performance.
5. The v0.9 dual-view branch lacked sufficient trustworthy `ow` source-mask coverage.
6. The system is a research prototype and requires expert review for every positive candidate.
7. A black final mask on a blocked scene means “no operational decision,” not proof that no oil exists.
8. `LOOK_ALIKE` and `UNCERTAIN` are both excluded from the final research mask.

9. Manual exploratory checks on known positive DARTIS scenes revealed false negatives. An empty candidate mask must be interpreted as “no high-confidence candidate was produced,” not as proof that oil is absent.
