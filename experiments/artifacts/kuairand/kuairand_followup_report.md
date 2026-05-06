# KuaiRand Follow-up Analyses

## Default Summary with 95% bootstrap intervals

| phase | detector | rate | ci_low | ci_high | median_delay | healthy_fp_per_user |
| --- | --- | --- | --- | --- | --- | --- |
| bubble_detection | CI | 0.458 | 0.409 | 0.512 | 25.0 | 0.749 |
| bubble_detection | CI^E | 0.692 | 0.646 | 0.741 | 31.8 | 1.387 |
| bubble_detection | CI^E-EWMA | 0.529 | 0.477 | 0.58 | 35.5 | 0.956 |
| bubble_detection | ADWIN | 0.0 | 0.0 | 0.0 | NA | 0.0 |
| bubble_detection | PageHinkley | 0.0 | 0.0 | 0.0 | NA | 0.0 |
| bubble_detection | KSWIN | 0.619 | 0.569 | 0.665 | 23.0 | 1.308 |
| bubble_detection | NoDrift | 0.0 | 0.0 | 0.0 | NA | 0.0 |
| collapse_detection | CI | 0.548 | 0.499 | 0.6 | 37.0 | 0.749 |
| collapse_detection | CI^E | 0.7 | 0.651 | 0.749 | 45.0 | 1.387 |
| collapse_detection | CI^E-EWMA | 0.567 | 0.515 | 0.619 | 41.8 | 0.956 |
| collapse_detection | ADWIN | 0.0 | 0.0 | 0.0 | NA | 0.0 |
| collapse_detection | PageHinkley | 0.0 | 0.0 | 0.0 | NA | 0.0 |
| collapse_detection | KSWIN | 0.978 | 0.962 | 0.992 | 26.5 | 1.308 |
| collapse_detection | NoDrift | 0.0 | 0.0 | 0.0 | NA | 0.0 |

## Paired improvement of CI^E over CI

| phase | comparison | delta_rate | ci_low | ci_high | p_value |
| --- | --- | --- | --- | --- | --- |
| bubble_detection | CI^E - CI | 0.234 | 0.183 | 0.292 | 0.0 |
| collapse_detection | CI^E - CI | 0.153 | 0.098 | 0.207 | 0.0 |

## Lambda sensitivity for CI^E

| lambda | phase | proxy | e0_scale | rate | ci_low | ci_high | healthy_fp_per_user |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | bubble_detection | kl | 1.0 | 0.463 | 0.414 | 0.512 | 0.777 |
| 0.5 | collapse_detection | kl | 1.0 | 0.55 | 0.499 | 0.602 | 0.777 |
| 1.0 | bubble_detection | kl | 1.0 | 0.474 | 0.422 | 0.526 | 0.837 |
| 1.0 | collapse_detection | kl | 1.0 | 0.559 | 0.507 | 0.61 | 0.837 |
| 2.0 | bubble_detection | kl | 1.0 | 0.575 | 0.523 | 0.627 | 1.215 |
| 2.0 | collapse_detection | kl | 1.0 | 0.629 | 0.578 | 0.678 | 1.215 |
| 3.0 | bubble_detection | kl | 1.0 | 0.692 | 0.646 | 0.741 | 1.387 |
| 3.0 | collapse_detection | kl | 1.0 | 0.7 | 0.651 | 0.749 | 1.387 |
| 4.0 | bubble_detection | kl | 1.0 | 0.747 | 0.7 | 0.79 | 1.632 |
| 4.0 | collapse_detection | kl | 1.0 | 0.752 | 0.706 | 0.796 | 1.632 |

## E0 sensitivity for CI^E

| e0_scale | phase | proxy | lambda | rate | ci_low | ci_high | healthy_fp_per_user |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | bubble_detection | kl | 3.0 | 0.845 | 0.807 | 0.88 | 1.837 |
| 0.5 | collapse_detection | kl | 3.0 | 0.796 | 0.755 | 0.837 | 1.837 |
| 1.0 | bubble_detection | kl | 3.0 | 0.692 | 0.646 | 0.738 | 1.387 |
| 1.0 | collapse_detection | kl | 3.0 | 0.7 | 0.651 | 0.747 | 1.387 |
| 2.0 | bubble_detection | kl | 3.0 | 0.529 | 0.477 | 0.58 | 1.019 |
| 2.0 | collapse_detection | kl | 3.0 | 0.591 | 0.537 | 0.643 | 1.019 |

## Effort proxy sensitivity for CI^E

| proxy | phase | lambda | e0_scale | rate | ci_low | ci_high | healthy_fp_per_user |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kl | bubble_detection | 3.0 | 1.0 | 0.692 | 0.646 | 0.741 | 1.387 |
| kl | collapse_detection | 3.0 | 1.0 | 0.7 | 0.651 | 0.749 | 1.387 |
| tv | bubble_detection | 3.0 | 1.0 | 0.569 | 0.52 | 0.619 | 1.014 |
| tv | collapse_detection | 3.0 | 1.0 | 0.613 | 0.564 | 0.662 | 1.014 |
| gini | bubble_detection | 3.0 | 1.0 | 0.556 | 0.504 | 0.605 | 1.008 |
| gini | collapse_detection | 3.0 | 1.0 | 0.629 | 0.58 | 0.676 | 1.008 |

## Threshold quantile sensitivity

| threshold_quantile | detector | phase | rate | ci_low | ci_high | healthy_fp_per_user |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | CI | bubble_detection | 0.286 | 0.24 | 0.332 | 0.409 |
| 0.1 | CI | collapse_detection | 0.341 | 0.292 | 0.384 | 0.409 |
| 0.1 | CI^E | bubble_detection | 0.485 | 0.439 | 0.537 | 0.916 |
| 0.1 | CI^E | collapse_detection | 0.45 | 0.398 | 0.501 | 0.916 |
| 0.1 | CI^E-EWMA | bubble_detection | 0.365 | 0.316 | 0.414 | 0.646 |
| 0.1 | CI^E-EWMA | collapse_detection | 0.327 | 0.281 | 0.376 | 0.646 |
| 0.2 | CI | bubble_detection | 0.458 | 0.406 | 0.51 | 0.749 |
| 0.2 | CI | collapse_detection | 0.548 | 0.499 | 0.597 | 0.749 |
| 0.2 | CI^E | bubble_detection | 0.692 | 0.646 | 0.738 | 1.387 |
| 0.2 | CI^E | collapse_detection | 0.7 | 0.651 | 0.744 | 1.387 |
| 0.2 | CI^E-EWMA | bubble_detection | 0.529 | 0.477 | 0.58 | 0.956 |
| 0.2 | CI^E-EWMA | collapse_detection | 0.567 | 0.515 | 0.619 | 0.956 |
| 0.3 | CI | bubble_detection | 0.553 | 0.501 | 0.602 | 0.932 |
| 0.3 | CI | collapse_detection | 0.689 | 0.643 | 0.736 | 0.932 |
| 0.3 | CI^E | bubble_detection | 0.785 | 0.744 | 0.826 | 1.747 |
| 0.3 | CI^E | collapse_detection | 0.82 | 0.779 | 0.858 | 1.747 |
| 0.3 | CI^E-EWMA | bubble_detection | 0.695 | 0.646 | 0.741 | 1.311 |
| 0.3 | CI^E-EWMA | collapse_detection | 0.763 | 0.722 | 0.804 | 1.311 |

## Tag-reweighted off-policy control

This control uses a clipped self-normalized tag-frequency reweighting signal.
It is a logged-data sanity check, not a causal IPS guarantee, because the
repository does not contain item-level propensities or replica policies.

| phase | detector | threshold | rate | ci_low | ci_high | healthy_fp_per_user | agreement_with_ci_e | spearman_min_score_vs_ci_e |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble_detection | tag_snips_proxy | 0.843 | 0.52 | 0.469 | 0.572 | 0.768 | 0.512 | 0.107 |
| collapse_detection | tag_snips_proxy | 0.843 | 0.534 | 0.482 | 0.586 | 0.768 | 0.556 | 0.021 |

## Downstream consequences after bubble flags

These rows compare collapse-phase outcomes for users flagged early versus
users not flagged early. They are not causal welfare estimates, but they do
test whether early `CI^E` alarms line up with worse downstream behavior.

| comparison | outcome | left_mean | right_mean | delta | ci_low | ci_high | n_left | n_right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CI flagged vs not | collapse_watch_ratio | 77.32 | 76.673 | 0.647 | -59.393 | 49.36 | 168 | 199 |
| CI flagged vs not | collapse_like_rate | 0.011 | 0.006 | 0.005 | -0.002 | 0.014 | 168 | 199 |
| CI flagged vs not | collapse_long_view_rate | 0.128 | 0.136 | -0.008 | -0.031 | 0.017 | 168 | 199 |
| CI flagged vs not | collapse_tag_entropy | 0.948 | 0.94 | 0.008 | 0.002 | 0.013 | 168 | 199 |
| CI^E flagged vs not | collapse_watch_ratio | 76.16 | 78.788 | -2.628 | -90.219 | 54.776 | 254 | 113 |
| CI^E flagged vs not | collapse_like_rate | 0.008 | 0.011 | -0.003 | -0.013 | 0.005 | 254 | 113 |
| CI^E flagged vs not | collapse_long_view_rate | 0.135 | 0.127 | 0.008 | -0.019 | 0.032 | 254 | 113 |
| CI^E flagged vs not | collapse_tag_entropy | 0.945 | 0.941 | 0.003 | -0.003 | 0.01 | 254 | 113 |
| CI^E-only vs neither | collapse_watch_ratio | 66.655 | 88.323 | -21.668 | -128.542 | 51.957 | 107 | 92 |
| CI^E-only vs neither | collapse_like_rate | 0.004 | 0.009 | -0.005 | -0.016 | 0.003 | 107 | 92 |
| CI^E-only vs neither | collapse_long_view_rate | 0.137 | 0.135 | 0.002 | -0.031 | 0.035 | 107 | 92 |
| CI^E-only vs neither | collapse_tag_entropy | 0.941 | 0.939 | 0.003 | -0.006 | 0.011 | 107 | 92 |
