# Revision Audit

## Flagship Summary

- Passive real-world streams still favor generic drift detectors, but the ranking depends on whether one values raw sensitivity or balanced precision: ELEC2 most sensitive = cusum (96 leads, precision 0.121), most precise = adwin (0.474); Bikes most sensitive = cusum (384 leads, precision 0.065), most precise = adwin (0.524).
- The distinctive surviving contribution is intervention-aware monitoring: on KuaiRand bubble detection rises from CI 0.458 to CI^E 0.692, with paired bootstrap delta 0.234 [0.183, 0.292].
- The synthetic masking benchmark shows the same pattern: at particle influence 0.3 and lambda 3.0, CI stays at 0.949 while CI^E drops to 0.697, creating masking gap 0.252; in the passive control, the gap is 0.000.

## Calibration and Sensitivity

- KuaiRand bubble detection is highest at lambda 4.0 (rate 0.747, healthy FP/user 1.632), but the current text's stable operating region [2, 3] is consistent with the measured trade-off.
- Stable-region rows:
  - lambda 2.0: bubble rate 0.575, healthy FP/user 1.215.
  - lambda 3.0: bubble rate 0.692, healthy FP/user 1.387.
- The strongest logged effort proxy remains KL for bubble detection (rate 0.692, healthy FP/user 1.387).

## Sinkhorn Runtime / Bias Trade-off

- At d=8, n=100, increasing epsilon from 0.05 to 1.0 changes runtime from 7.341 ms to 0.661 ms and mean abs. bias from 0.6102 to 0.1383.
- At d=256, n=100, the same change moves runtime from 0.799 ms to 0.571 ms and mean abs. bias from 1.7778 to 0.1446.
- This confirms the current manuscript's interpretation: epsilon is an operational calibration knob, not a universal fixed setting.

## Immediate Revision Priorities

1. Promote CI / CI^E and coercive masking as the flagship contribution in the abstract, introduction, and conclusion.
2. Keep the cube-root law as the rigorous backbone, but explicitly state that passive streams remain ADWIN-favored in the current evidence bundle.
3. Move the fairness/calibration message for KuaiRand into a more prominent sentence in the body text: healthy-only thresholds, same scalar input, paired bootstrap intervals.
4. Tighten the Sinkhorn citation chain around the null-vs-alternative calibration split and published Goldfeld et al. citation.
5. Strengthen the lower-bound proposition by making the operational constant/regime explicit rather than only saying "universal constant".

## Reviewer-Facing Takeaway

- KuaiRand CI^E bubble/collapse rates: 0.692 / 0.700.
- KuaiRand strongest generic passive baseline on bubble onset: KSWIN 0.619.
- KuaiRand paired improvements over CI: bubble 0.234, collapse 0.153.
- Particle masking gap at default setting: 0.252.
