# Review Response Report

## Purpose

This report is the shared reference for all follow-up phases. It records what the review got right, what must be corrected in the paper, what evidence already exists, and what still needs new proofs, ablations, or baselines.

## Executive Decision

We will pursue the strongest defensible version of the contribution, even if that requires new proofs and new ablations. The paper must reflect evidence, not defend a claim after the evidence weakens it.

The project remains centered on two flagship ideas:

1. finite-memory tracking under drift has a real coherence-delay floor;
2. intervention-aware diagnostics can expose coercive masking in action-coupled systems.

## Review Questions and Answers

### 1. How general is the cube-root law?

Current answer: the clean derivation is rigorous for the finite-memory averaging class already written in the paper, with sliding windows and EWMA as the verified cases. The paper now also proves a minimax lower bound for arbitrary measurable estimators restricted to the last $m$ samples on a discrete $\zeta$-Lipschitz drift class, which recovers the same cube-root exponent in the critical-window regime. The current text would still be too broad if it suggested a universal minimax statement over all recursive or extrapolative adaptive estimators.

What we will do:
- either prove a formal minimax lower bound for an explicit class of finite-memory kernels, or
- narrow the theorem language to a precise estimator class if the stronger proof does not close.

Preferred route:
- prove a lower bound for a class of causal, finite-memory averaging kernels with effective size and effective lag parameters;
- then broaden the lower-bound side to arbitrary window-limited estimators on a Lipschitz drift class if the proof closes cleanly.

Status update:
- the paper includes a kernel-level lower bound for the causal averaging family, so the cube-root law is exact for the class actually studied;
- it now also includes a minimax lower bound for arbitrary measurable estimators with access only to the last $m$ samples of a discrete $\zeta$-Lipschitz drift path, which broadens the theorem beyond linear averaging and recovers the same cube-root exponent in the critical-window regime without claiming universality for all adaptive estimators.

### 2. Is there a sign error in Proposition 6.12?

Current answer: the live source states the positive cube-root scaling, and the manuscript/PDF audit is complete.

What we will do:
- no further errata note is needed unless a stale copy reappears in a future artifact.

Audit result:
- the manuscript and compiled PDF are aligned on the positive cube-root scaling;
- no stale `\zeta^{-1/3}` remnants remain in the live corpus.

### 3. How far does the ISS template go?

Current answer: the Borromean stability argument is a conditional template, constructively verified in the Gaussian case. It is not yet an operational theorem for arbitrary black-box `\Phi`.

What we will do:
- keep the template framing honest,
- strengthen the Gaussian instantiation if possible,
- explicitly separate theorem, template, and future work in the paper.

### 4. Are the action-argument necessity claims too strong?

Current answer: yes, the “must depend on three arguments” framing is stronger than necessary and risks sounding more novel than it is.

What we will do:
- map the discussion more directly to standard belief-state / POMDP / filtering language,
- keep the three-role decomposition as a modeling choice, not a universal ontological claim.

### 5. How robust is the effort-aware score?

Current answer: the score is promising, but its behavior under proxy mis-specification, `\lambda`, and `E0` is not yet fully pinned down.

What we will do:
- expand sensitivity and ablation analyses,
- treat `\sigma_A` as exact only in deterministic or constructively specified settings, and as a surrogate elsewhere,
- add explicit proxy-quality criteria.

Current ablation evidence:
- the particle and KuaiRand `\lambda` sweeps already show the stable operating range,
- the domain-specific `E0` choices are documented in the paper,
- exact / identified / proxy semantics are separated in the paper and protocol appendix.

Operational ladder:
- `exact`: deterministic action channels or constructively specified replicas.
- `identified`: settings where the action channel is directly logged well enough to estimate the target quantity with a tractable estimator.
- `proxy`: logged or observational settings where the target is only approximated by an operational surrogate.

### 6. How honest is the KuaiRand interpretation?

Current answer: it is useful, but off-policy confounding and exposure bias remain real limitations.

What we will do:
- tighten the language around logged-data proxies,
- add stronger controls and matched comparisons,
- avoid causal claims that the data cannot support.

Current operational stance:
- KuaiRand is a logged-data surrogate benchmark, not a causal identification study.
- The reported score is useful because it tracks exposure concentration and masking, not because it equals the unobserved conditional entropy exactly.
- `\sigma_A` in KuaiRand belongs to the proxy level of the ladder.

### 7. Are there enough baselines?

Current answer: largely resolved for the current passive benchmark set.

What we will do:
- CUSUM, forgetting-factor RLS, and scalar Kalman baselines are now added on
  ELEC2 and Bikes under the same matching discipline;
- benchmark any new baseline under the same calibration discipline.

Current evidence:
- ADWIN remains the strongest generic passive detector on both ELEC2 and Bikes;
- CUSUM is sensitive but imprecise;
- FF-RLS and Kalman sit between the fixed-window score and ADWIN, but do not
  overturn the passive-stream ordering;
- the Fr\'echet baseline is useful on Bikes but inactive on ELEC2 at the current
  calibration.

### 8. What about Sinkhorn runtime and practical cost?

Current answer: the runtime cost is now measured on the NumPy Sinkhorn prototype.

What we will do:
- keep the runtime/bias trade-off explicit in the manuscript;
- record the measured throughput and latency as prototype-level evidence;
- distinguish theoretical rate from deployed runtime.

Current evidence:
- runtime drops sharply as regularization increases;
- window size increases cost in the expected direction;
- the measured cost should be read as a prototype profile, not a deployment ceiling.

### 9. Should we rename the diagnostics and the project?

Current answer: yes, but only after the review-response work is stabilized.

Preferred naming direction:
- `C` -> `CI` (`Coherence Index`)
- `C^E` -> `CI^E` (`Effort-Adjusted Coherence Index`)
- `M_t` -> `Masking Gap`
- umbrella: tracking/diagnostic stack centered on `CI` and `CI^E`

We will not force a rename if it weakens the paper or obscures the math.

## Evidence Already Available

- sliding-window cube-root slope is already near `1/3`
- EWMA also shows the same scaling
- particle masking clearly separates apparent coherence from effort-aware coherence
- KuaiRand shows the logged/active regime where effort-aware diagnostics matter
- passive benchmarks still favor generic drift detectors like ADWIN

## Main Gaps That Still Need Work

- deployment-grade optimization of the Sinkhorn pipeline
- stronger controls for KuaiRand / logged data

## Implementation Rule

Every future phase must update the paper if the evidence changes the claim.

## Round-2 Follow-up Answers

This section records the follow-up analyses run after the later review pass. The
main artifact bundle lives under `experiments/artifacts/kuairand/` in:

- `kuairand_followup_report.md`
- `kuairand_followup_default.csv`
- `kuairand_followup_improvement.csv`
- `kuairand_followup_lambda.csv`
- `kuairand_followup_e0.csv`
- `kuairand_followup_proxy.csv`
- `kuairand_followup_threshold.csv`
- `kuairand_followup_offpolicy.csv`
- `kuairand_followup_downstream.csv`

### Q1. How robust are the `CI^E` results to the effort proxy and to `\lambda` / `E_0` calibration?

Answer:
- On KuaiRand, the `CI^E` improvement over `CI` survives healthy-only threshold
  calibration and remains statistically separated at the default operating
  point. Bubble detection rises from `0.458` for `CI` to `0.692` for `CI^E`,
  with a paired improvement of `+0.234` and bootstrap interval
  `[0.183, 0.292]`.
- The `\lambda` sweep keeps the same monotone pattern already seen in the paper:
  stronger effort penalties increase bubble/collapse sensitivity but also raise
  healthy false positives per user.
- The `E_0` sweep shows the benchmark is not calibration-free. Halving `E_0`
  pushes bubble detection to `0.845` but raises healthy false positives to
  `1.837/user`; doubling `E_0` lowers bubble detection to `0.529` and healthy
  false positives to `1.019/user`. The present default remains a compromise,
  not a uniquely identified optimum.
- Alternative effort proxies were tested at the same default operating point.
  KL remains the strongest of the three proxies available in this repository:
  TV and Gini-style concentration gaps still improve over plain `CI` in some
  settings but sit materially below the default KL proxy.

Bottom line:
- `CI^E` is robust enough to survive proxy and calibration perturbations, but
  the operating point matters and should be reported explicitly rather than
  treated as universal.

### Q2. Can we compare `CI^E` against causal / counterfactual baselines or off-policy risk estimates?

Answer:
- Not faithfully with the current repository inputs. We do not have item-level
  propensities, replica policies, or the logged interventions needed to run a
  CAFL/CRM-style counterfactual benchmark honestly.
- To avoid pretending otherwise, we added only a coarse logged-data control:
  a clipped self-normalized tag-frequency reweighting signal
  (`kuairand_followup_offpolicy.csv`).
- That control detects some later-phase change (`0.520` bubble,
  `0.534` collapse) but aligns only weakly with `CI^E`
  (`agreement 0.512/0.556`, very small score correlations).

Bottom line:
- This follow-up does not close the causal-validation gap. It confirms that the
  paper still lacks a falsifiable counterfactual benchmark for masking, exactly
  as the reviewer suspected.

### Q3. Beyond averaging families, can model-based predictors beat the `\zeta^{1/3}` floor?

Answer:
- Under the current `\zeta`-Lipschitz drift assumption alone, our lower-bound
  story applies to the finite-memory/window-restricted regime actually studied.
- Extrapolative or model-based predictors can only beat that floor if they
  exploit stronger structure than the current theorem assumes, for example:
  identifiable state dynamics, higher-order temporal smoothness, parametric
  trend structure, or external latent-state supervision.
- The present paper still does not prove a minimax lower bound over that richer
  model-based class, so the honest answer remains conditional: stronger models
  may beat the current floor, but only by importing stronger regularity or
  identifiability assumptions.

Bottom line:
- The cube-root law remains exact for the averaging family and relevant for
  window-limited estimators, not universal for all adaptive predictors.

### Q4. What do we now know about Sinkhorn dimension dependence and regularization bias?

Answer:
- The existing Gaussian runtime artifact already answers part of this question:
  `experiments/artifacts/gaussian/gaussian_sinkhorn_runtime.csv` measures
  dimensions `d in {2, 8, 32}` at several windows and regularization levels.
- The practical picture is consistent with the manuscript wording: low
  regularization amplifies bias in higher dimension, while larger
  regularization sharply lowers runtime and keeps bias operationally moderate
  on the validated grid.
- What we still do not have is the reviewer's stronger test: a real,
  high-dimensional embedding benchmark in the hundreds of dimensions.

Bottom line:
- We now have a measured dimension/bias profile on controlled Gaussian clouds,
  but not yet the high-dimensional real benchmark that would truly pressure-
  test the framework.

### Q5. How fair is threshold calibration, and are the main KuaiRand gains statistically significant?

Answer:
- We now report the actual healthy false positives per user for all detectors at
  the chosen operating points instead of only headline detection rates.
- `CI`, `CI^E`, and `CI^E`-EWMA are calibrated on healthy windows only using the
  same quantile rule; threshold sensitivity was re-run for quantiles
  `{0.1, 0.2, 0.3}`.
- At the default `0.2` quantile, `CI^E` remains above `CI` with paired bootstrap
  intervals and exact paired-significance checks:
  bubble `+0.234 [0.183, 0.292]`, collapse `+0.153 [0.098, 0.207]`.
- The passive baselines still occupy different effective false-alarm regimes.
  For example, KSWIN is highly sensitive on collapse but already sits at a high
  healthy false-positive rate in the same benchmark.

Bottom line:
- The `CI^E` gain over `CI` is statistically real on this benchmark, but the
  overall detector comparison is still calibration-sensitive rather than fully
  equalized by false-alarm rate.

### Q6. What practical guidance can we now give for estimating `\sigma_A` and `\sigma_\Phi` in black-box systems?

Answer:
- The exact / identified / proxy ladder remains the right operational framing.
- The clearest practical recommendation that emerged from this pass is:
  if a shadow policy, replica policy, or direct intervention log exists, use it
  to estimate action-induced exposure distortion directly and treat that as the
  highest-quality route to `\sigma_A`.
- Without such logs, the benchmark falls back to proxy semantics and should be
  described as such. KuaiRand remains at the proxy level.
- For `\sigma_\Phi`, the repository still has no general black-box estimator.
  The honest guidance is limited to surrogate stability diagnostics:
  repeated-seed replay, logged update norms, or explicit simulator-side energy
  functions when those are available.

Bottom line:
- The follow-up did not solve black-box identifiability of `\sigma_A` or
  `\sigma_\Phi`; it clarified that shadow-policy style logging is the most
  credible next step if we want to move beyond proxies.

### Q7. Do early `CI^E` flags align with downstream user consequences in KuaiRand?

Answer:
- We ran an external-outcome check using collapse-phase tag diversity,
  `long_view`, and `like` rates (`kuairand_followup_downstream.csv`).
- The current logged benchmark does not show a clean downstream-separation story
  for `CI^E`. Some differences move in the expected direction, but the evidence
  is weak and inconsistent on the available observables.
- This is an informative non-result. It means the current KuaiRand benchmark is
  still better at demonstrating intervention-aware sensitivity than at tying the
  score to an externally validated welfare consequence.

Bottom line:
- The strongest next benchmark should include either counterfactual logs,
  shadow-policy observations, or an external human/welfare label. The present
  repository does not yet have that surface.

## Commands Run For The Follow-up

From `experiments/`:

```bash
uv sync
uv run python run.py kuairand_followup
uv run python run.py kuairand --artifacts-dir ./artifacts/kuairand
```

## Rebuttal Draft For The New Review

The rebuttal should lead with the new uncertainty quantification, then narrow the
logged-data claim before the reviewer has to ask again, and finally turn the
proxy comparison into a concrete practitioner-facing result.

### Core paragraph

We thank the reviewer for asking for uncertainty quantification. We have now
added paired bootstrap intervals for the KuaiRand comparison. At the default
healthy-only calibration, $CI^E$ improves over $CI$ by $+0.234$ on bubble
detection (95\% bootstrap CI $[0.183, 0.292]$) and by $+0.153$ on collapse
detection (95\% bootstrap CI $[0.098, 0.207]$). These intervals exclude zero,
so the main empirical gain is statistically stable on this benchmark rather
than a small-sample fluctuation.

### Causal-ceiling paragraph

We also agree with the reviewer that the current KuaiRand benchmark provides
robustness evidence, not causal identification of masking. We have made this
boundary explicit in the manuscript and protocol appendix: the logged benchmark
is presented as proxy-level evidence for effort-aware monitoring, not as a
counterfactual estimate of intervention-induced misalignment. In the current
repository, a shadow-policy benchmark, replica-policy logs, or external
human-grounded outcomes would be required for the stronger causal claim, and we
now identify that as the primary open direction rather than implying that the
present logged benchmark closes it.

### Proxy-selection paragraph

We expanded the effort-proxy sensitivity analysis at the same operating point.
Among the proxies available in this repository, the KL-based effort proxy is the
strongest default: it yields higher bubble and collapse detection than the TV-
and Gini-based alternatives under the same healthy-only calibration. We view
this as a useful practical result in its own right, because the reviewer is
correct that proxy choice matters whenever exact $\sigma_A$ is unavailable.

### Optional closing paragraph

Taken together, the new analyses sharpen the empirical claim without changing
its scope. The paper now supports three narrower but stronger statements: (i)
the KuaiRand gain of $CI^E$ over $CI$ is statistically stable on the logged
benchmark, (ii) the logged benchmark should be read as robustness evidence for
effort-aware diagnostics rather than causal identification of masking, and (iii)
KL is the strongest effort proxy among the proxy families we can test here.

### Evidence references to cite in the rebuttal

- Statistical intervals and sensitivity tables:
  `experiments/artifacts/kuairand/kuairand_followup_report.md`
- Main KuaiRand benchmark wording:
  `theory/empirical_validation.tex` (`Experiment E: Flagship Real-Data Masking on KuaiRand`)
- Logged-data limitation wording:
  `discussion/limitations.tex`
- Proxy-level protocol wording:
  `appendices/kuairand_protocol.tex`

## Round-3 Follow-up: Second Review Response (Phase 18)

### Review verdict

The second review recommends acceptance after clarifications and empirical
strengthening. The reviewer identifies the cube-root law and effort-aware
diagnostics as the flagship contributions, and the empirical slopes matching
theory as particularly convincing.

### Actions taken

1. **Citations added:**
   - Strategic classification: `\cite{HardtMPW2016}` — connected in
     `discussion/related_work.tex` to the coercive masking framing.
   - Kernel MMD two-sample test: `\cite{GrettonEtAl2012}` — new
     paragraph in `discussion/related_work.tex` discussing the relationship
     between MMD and $W_2$.
   - Intrinsic-dimension OT rates: `\cite{NilesWeedBach2019}` — cited
     in the transport-based diagnostics paragraph to inform $C_K$ constant
     interpretation and $\varepsilon$ tuning guidance.

2. **Kernel MMD baseline implemented and run:**
   - `experiments/core/baselines.py`: `run_mmd_detector()` using unbiased
     $\text{MMD}^2$ with Gaussian RBF kernel, median-heuristic bandwidth,
     prefix-subsampled reference, and strided evaluation.
   - ELEC2: 131 warnings, 48 leads, 37% precision, median lead 74.
   - Bikes: 152 warnings, 75 leads, 49% precision, median lead 22.
   - Both tables in `theory/empirical_validation.tex` updated.
   - ADWIN remains the lead-count leader on both streams, but MMD is
     competitive on precision (especially Bikes) and provides the kernel
     two-sample test comparison the reviewer requested.

3. **Manuscript updated:**
   - Tables 9 and 11 now include the MMD row.
   - Prose in Experiments D and G discusses MMD results.
   - `discussion/limitations.tex` updated to reflect MMD is no longer missing.
   - `discussion/related_work.tex` expanded with kernel-shift and
     strategic-classification paragraphs.
   - Manuscript recompiled cleanly with `tectonic main.tex`.

4. **High-dimensional Sinkhorn evaluated ($d \gg 32$):**
   - The Gaussian tracker validation was extended to $d \in \{64, 128\}$.
   - At $d=128$, the Sinkhorn solver remains highly efficient: for $n=100$
     and $\varepsilon=1.0$, the solver processes windows in under 0.6\,ms
     with a mean bias of 0.14.
   - Table 7 and \Cref{fig:gaussian_sinkhorn_runtime} have been updated.

### Remaining open items from this review

- **Theoretical detection-power guarantee for CI^E:** The reviewer asks
  whether CI^E can be given any formal guarantee (e.g., false-positive
  control under a linear-Gaussian performative model). This has not been
  attempted. The Gaussian tracker is the natural setting for such a result.
- **Smoothed-min aggregation:** The reviewer suggests smoothed or
  confidence-adjusted alternatives to the min aggregation. We acknowledge
  this as a legitimate design-space question; the current min is justified
  by the Borromean property.

