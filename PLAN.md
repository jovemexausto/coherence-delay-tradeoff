# PLAN

## Status

O objeto central é a **validade temporal sob drift**.

- **validade temporal sob drift**
- **horizonte prospectivo de validade**
- **inferência bootstrap-calibrada do horizonte**
- **controle operacional de memória**
- **custo explícito de atuação** como fecho da camada operacional

O arco científico consolidado é:

- lei constitutiva
- inferência do horizonte a partir da geometria de lag
- calibração robusta da incerteza
- política operacional de memória com custo explícito de atuação

As principais descobertas empíricas são:

- `H`, `zeta` e `n^*` são observáveis via lag geometry sob o modelo correto
- misspecification de forma aparece em diagnósticos locais
- heterocedasticidade muda cobertura/thresholds mais do que o ponto estimado
- o controller só separa de forma clara quando memória vira estado controlado com custo explícito
- o mapa de fases em `(H, lambda_0, lambda_1)` é regime-dependente; não existe política universalmente dominante
- `signed_residual` não melhora o gap validade-detecção e parece piorar a taxa de gap positivo no frontier reduzido
- a razão de particionamento `R_t(n)` é uma coordenada dual candidata: em `n^*` ela vale `a/H`, mas fora do benchmark sua utilidade depende da qualidade da decomposição observada
- a dualidade do horizonte fica mais forte quando `R_t(n)` é usada como coordenada de sinal: ela identifica o lado de `n^*` sem estimar `(a,H,\zeta)` e supera a recuperação de slope em grid esparso
- o harness `code/useful_memory_horizon/hypothesis_suite.py` agora testa H1-H23 com benchmarks de controller, delay, ratio, misspecification e U-curve model-free
- `code/useful_memory_horizon/partition_ratio.py` é agora o objeto dual central: ratio exata, monotonicidade e inversão de horizonte entram como primitives do núcleo
- em benchmark sintético rápido, o controlador monotônico local supera levemente o two-point self-normalized step, então ele é o candidato default online sem conhecer `H`
- `code/useful_memory_horizon/ratio_control.py` adiciona um processo persistente com smoothing/deadband; em ruído alto ele supera a política instantânea e reduz updates, o que o torna a camada online mais importante do objeto dual
- `hypothesis_suite.py` agora fecha o crossover do controlador com H24-H25: persistência vence em ruído alto e instantâneo vence em ruído baixo
- o benchmark comparativo mais amplo sugere mapa de regimes, não vencedor universal: `instant_ratio` domina a maior parte das células, `persistent_ratio` ocupa um subconjunto ruidoso e `lag_geometry` ainda vence em algumas células de `H` alto
- o proxy-based regime router ainda não é competitivo no grid amplo; o gargalo parece ser sensing do regime, não a regra de política em si
- `regime_route_delay.py` mede `\Delta_{reg-route}`; o atraso de roteamento é positivo mesmo com sensing perfeito e cresce com ruído, confirmando que o meta-controlador herda coerência-atraso
- `meta_sensing_benchmark.py` mostra que um ensemble multiescala melhora o roteamento em ruído moderado, mas satura em ruído alto; sensing pode melhorar a fronteira, não apagar o atraso estrutural
- `policy_frontier_theorem.py` gera a figura central da fronteira de políticas em `artifacts/figures/policy_frontier/`; o sensor oracular já tem atraso positivo e o ensemble multiescala só reduz a parte evitável em ruído moderado
- a figura principal de sensing já está gerada em `artifacts/figures/meta_sensing/fig_meta_sensing_frontier.{pdf,png}` com tabela de suporte em `artifacts/tables/meta_sensing/meta_sensing_summary.csv`

Os spin-offs existem só como superfícies de migração e não como narrativas independentes.

## Core Thesis

O paper raiz deve afirmar uma única tese forte:

> Sob drift, a persistência inferencial é governada por uma lei constitutiva de validade temporal que induz um horizonte ótimo de memória, permite inferência calibrada a partir da geometria de lag e fecha operacionalmente em uma política de controle de memória que equilibra perda de validade e custo explícito de atuação.

Conjectura operacional latente:

`\mathcal L_t(n_t^\pi) + \alpha E_t + \beta \Delta_{val-det} \ge \gamma`


Essa desigualdade não é theorem-level; ela organiza o mapa de fases e a camada operacional.

Em termos formais, o objeto principal permanece:

`E d(\widehat P_t^{(n)}, P_t) <= C_K n^{-a} + C_S \zeta n^H`

com horizonte induzido

`n^*(a,H) ~ (C_K / \zeta)^{1/(a+H)}`.

Tudo o que entra no manuscrito final deve servir diretamente a essa tese.

Como framing auxiliar da camada operacional, a parte sequencial pode ser lida como
uma tríade **P--A--Φ**:

- **P (Potencialidade)**: a região de validade ainda disponível, resumida pelo horizonte
  e pela banda near-optimal de memória
- **A (Atuação)**: a política que age sobre a memória ao longo do tempo e paga custo explícito de reconfiguração
- **Φ (Convergência)**: o funcional que mede quão bem a atuação mantém a memória
  realizada próxima da região temporalmente válida, penalizando perda de validade e esforço de atuação

Essa tríade não compete com o objeto principal; ela descreve o fecho operacional do horizonte.

O custo explícito de atuação é

`c_t^pi = lambda_0 1{n_t^pi != n_{t-1}^pi} + lambda_1 |log n_t^pi - log n_{t-1}^pi|`

e o objeto operacional passa a ser a soma entre perda de validade e esforço de atuação.

## Scientific Claims

### Closed / strong claims

1. Existe uma lei constitutiva do horizonte de validade temporal sob drift.
2. O horizonte `n^*` é imposto pela geometria do problema, não por tuning contingente.
3. `H`, `zeta` e `n^*` são inferíveis a partir da geometria de lag sob o modelo correto.
4. Heterocedasticidade nos regimes testados não move materialmente o ponto estimado de `H` e `n^*`, mas exige bootstrap mais honesto.
5. O controller só se separa de forma clara quando a memória é um estado controlado com custo explícito de atuação.
6. Em regimes fortes, o mapa de fases em `H` depende de custo e drift; não existe política universalmente dominante.
7. `signed_residual` não melhora o frontier reduzido; `observation` segue como melhor baseline.

### Negative results that matter

1. `variance_bridge` não justificou promoção da modelagem explícita de variância ao núcleo do estimador.
2. Detectores sozinhos não resolvem o problema operacional; é a política de memória que importa.
3. A claim forte de suficiência pre-drift para PHH não se sustentou em dados reais.
4. `signed_residual` é mais conservador e não corrige o gap validade-detecção.

### Open claims / still frontier

1. bound formal de regret do controller
2. integração teórica plena de `\tau_valid`, `\tau_detect`, `\Delta_{val-det}`
3. replay convincente em streams reais
4. teoria mais apertada para misspecification misto forma+escala

## Information Architecture

### Root paper

O `main.tex` na raiz deve ser o único manuscrito ativo.

Ele deve absorver a teoria do horizonte, a inferência por geometria de lag, a calibração bootstrap e o controller sequencial.

### Surviving code surface

O namespace de código sobrevivente, por enquanto, deve ser:

- `projects/temporalbridge/`

porque ele já encapsula:

- `fit_horizon`
- `bootstrap_horizon`
- `calibrate_alarms`
- `detect_alarms`
- `validity_controller`

e permite consolidar a linha operacional sem levar toda a história de experimentos para a raiz.

### Spin-offs to absorb, not preserve as separate stories

- `projects/scale-consistency/`
  - absorver resultados, tabelas, código estável e artefatos úteis
  - não preservar como narrativa científica separada

- `projects/sequential-validity-detection-delay/`
  - absorver `\tau_valid`, `\tau_detect`, `\Delta_{val-det}`
  - não preservar como paper paralelo

- `projects/hierarchical-adaptation-burden/` e linha PHH
  - útil como projeto vizinho e arquivo de resultados
  - não deve organizar o paper raiz atual

## Manuscript Arc

O arco do paper segue: constitutive law -> horizon inference -> calibration -> control -> empirical phase map.

### Arc 3. Calibration under misspecification and heteroskedasticity

- misspecification diagnostics
- bootstrap robusto
- resultado negativo sobre modelagem explícita de variância

### Arc 4. Operational closure

- memória como estado controlado
- controller
- custos de update, inércia, deadband e esforço de atuação
- benchmark sequencial
- validity loss, excess loss, actuation cost, regret, lead time

Essa é a narrativa. Tudo o que não servir diretamente a esse arco deve ir para apêndice, nota de projeto, ou ser congelado.

## Root Paper Structure

1. **Introduction**
2. **Constitutive Horizon Law**
3. **Horizon Inference from Lag Geometry**
4. **Calibration under Misspecification and Heteroskedasticity**
5. **Operational Memory Control under Drift**
6. **Sequential Benchmark**
7. **Real-stream Replay**
8. **Discussion and Limitations**
9. **Conclusion**

### What should stay in appendix

- full bootstrap tables
- full misspecification tables
- extra diagnostic plots
- controller ablations in full detail
- lambda-grid sweeps for actuation cost
- implementation-heavy details

## Recent Advances To Preserve In The Root Story

### Horizon bridge / inference line

- expanded E1/E2 grids
- lag-count aware reporting
- KL diagnostics
- robust bootstrap coverage experiment
- negative result on explicit variance modeling

### Temporalbridge / controller line

- thin, cleaner scientific API surface
- benchmark trio (`3/3`) then short grid (`7/7`)
- Monte Carlo action benchmark favoring controller over baselines
- sequential benchmark with full validity curve, memory dynamics, update cost, deadband and explicit actuation cost
- phase-map analysis artifacts for `default` and `strong` schedules
- notebook-style analysis surface writing reproducible artifacts

### Sequential delay line

- the two-clock framing is too important to discard:
  - validity clock
  - detection clock
- `\Delta_{val-det}` should enter the root paper as a metric and conceptual object, even if the spin-off itself is retired as an active manuscript.

## Engineering State

### In `projects/scale-consistency`

- stable tested backend for lag-geometry inference and bootstrap
- runner modes for smoke and bootstrap coverage
- bridge artifacts in CSV/JSON/PDF
- currently the most validated code surface for inferential machinery

### In `projects/temporalbridge`

- clean scientific facade on top of the validated backend
- controller and benchmarks
- notebook-style analysis script
- phase-map analysis script sweeping `H × \lambda_0 × \lambda_1`
- dedicated smoke tests

### Engineering principle from now on

- no more multiplying top-level scientific narratives
- no more new spin-offs unless they isolate a genuinely distinct object
- consolidate stable logic into `temporalbridge`
- keep root manuscript as the only paper actively optimized

## Current Frontier

These are the real next problems, in order.

1. **Consolidate the paper**
   - absorb spin-off ideas into root sections
   - rewrite framing around one object and one arc

2. **Controller ablations**
   - `wild` vs `moving_block`
    - `lambda_0` low/high
    - `lambda_1` low/high
    - tracking gain low/high
    - deadband / persistence
   - identificar o ponto de inversão onde custo fixo alto favorece detector-only em custo total

Recent sweep result:

- no tested grid, `controller` remains best overall on average, but `detector_only` wins in a subset of high-`lambda_0` cells
- the inversion boundary appears when fixed update cost is large enough; in the current sweep, `lambda_0 ~ 2` is sufficient for `detector_only` to undercut `controller` in total cost
- `deploy_only` never became the best non-oracle policy in the tested grids
- `controller` kept lower `log_memory_std` than `detector_only`, so the inversion is cost-driven rather than a stability advantage for `detector_only`
- current status: H4/H10 supported, H9 supported, H8 not supported, H7 mixed, H1-H3 and H6 still need explicit masking/delay instrumentation beyond the current proxy metrics
- with calibrated delay slack (`0.25`), `tau_valid` becomes nontrivial and `delay_gap` separates `controller` from `detector_only`: the controller detects before validity breaks, while detector-only lags behind the validity crossing and shows a larger masking index
- in `schedule_mode="strong"`, the no-update baseline can become the best total-cost policy once update cost is included, even though it is not the best-validity policy; this is the clearest current sign of an energy-effectiveness limit under severe drift
- across `H in [0.1, 0.9]`, the default schedule shifts toward controller dominance at higher `H`, while the strong schedule shifts toward detector-only / no-update dominance at higher `H`; this is the first clean phase-map signal in the `(H, \lambda_0, \lambda_1)` space

3. **Integrate the delay object explicitly**
   - make `\tau_valid`, `\tau_detect`, `\Delta_{val-det}` first-class in the sequential benchmark and writeup

4. **Real-stream replay**
   - ELEC2 / MOA-like streams first
   - then only harder domains if needed

5. **Theory sharpening**
    - tighter lower theory beyond the benchmark sign law
    - regret-style or validity-loss guarantees for the controller with actuation cost
    - formalize `Φ` as convergence functional for the operational layer

## Falseifiable Hypotheses

These hypotheses are written to be falsified by the sequential benchmark and the cost-grid sweep.

| Hypothesis | Falsified if |
| --- | --- |
| H1. Coerced action can mask validity loss | high `Φ` always coincides with in-band `\mathcal L_t(n)` |
| H2. Effort rises before validity collapses | `c_t^\pi` never leads validity degradation |
| H3. Delay exposes masking | `\Delta_{val-det}` is insensitive to masking regimes |
| H4. Cost regime changes ranking | the `\lambda_0,\lambda_1` grid never changes the best non-oracle policy |
| H5. Lead time can be masked | positive lead time always implies low validity loss |
| H6. Energy effectiveness has a limit | no high-drift regime forces collapse in total cost or validity |
| H7. Detector-only is less masked but less valid | detector-only behaves like controller on both masking and validity |
| H8. Deploy-only approaches controller under cost pressure | deploy-only never approaches controller in total cost at high cost |
| H9. Controller stability is visible in memory variance | `\operatorname{sd}(\log n_t^\pi)` does not separate policies |
| H10. Inversion boundary exists | `detector_only` never undercuts `controller` on any tested grid cell |

### Result Table Design

For each row `((\lambda_0, \lambda_1), \pi)`, report:

- `mean_cumulative_validity_loss`
- `mean_cumulative_update_cost`
- `mean_total_cost`
- `mean_regret`
- `mean_lead_time`
- `mean_log_memory_std`
- `mean_tau_valid`
- `mean_tau_detect`
- `mean_delay_gap`
- `mean_masking_index`

For each grid cell, also report:

- `best_non_oracle_policy`
- `best_non_oracle_total_cost`
- whether `controller` or `detector_only` wins on total cost

### Central Table Layout

| Policy | Validity loss | Actuation cost | Total cost | Regret | Lead time | logstd | tau_valid | tau_detect | delay_gap | masking_index |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controller | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric |
| detector_only | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric |
| deploy_only | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric |
| fixed_policy | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric | row metric |

Cell-level summary beneath the table:

- `best_non_oracle_policy`
- `best_non_oracle_total_cost`
- inversion flag: `controller` vs `detector_only`
- high-cost boundary indicator

## Limitations

1. Sequential theory is still weaker than the empirical controller results.
2. Real-stream evidence for the controller is not yet in place.
3. The current sequential schedule is informative, but still synthetic and hand-built.
4. Some project branches in the repo still reflect exploratory phases that should no longer dictate the main story.
5. `temporalbridge` still depends on code in `projects/scale-consistency`; migration is not complete.

## Latent Constitutive Law

The operational layer suggests a lower-bound structure of the form

`\mathcal L_t(n_t^\pi) + \alpha E_t + \beta \Delta_{val-det} \ge \gamma`

with `E_t` the actuation cost and `\Delta_{val-det}` the validity-detection lag.

Status:

- conjecture
- phase-map organizing principle
- not yet theorem-level

## Repository Consolidation Direction

### Active

- root manuscript
- `projects/temporalbridge`

### Source material to absorb and then retire from active scientific status

- `projects/scale-consistency`
- `projects/sequential-validity-detection-delay`

### Freeze / keep peripheral

- PHH burden line
- architecture-specific side analyses that do not serve the root object directly

## Immediate Next Steps

1. Rewrite the root manuscript outline around the three arcs:
    - constitutive law
    - inference and calibration
    - operational closure

2. Port stable code from `scale-consistency` into `temporalbridge` until dependency becomes optional rather than structural.

3. Add controller ablation runners and tables to the notebook-style analysis surface.
4. Add explicit `lambda_0`/`lambda_1` sweeps to the sequential benchmark outputs.

5. Integrate `\Delta_{val-det}` into the sequential benchmark outputs.

6. Run short real-stream replay before any final paper freeze.

## Bottom Line

The project is now coherent.

We are no longer trying to understand what the contribution is.
We now know it:

> a constitutive law of temporal validity under drift, an inferential pipeline that recovers the induced horizon with honest uncertainty, and a memory controller that uses that horizon to make better sequential decisions than detector-only or naive baselines.

The operational version of that claim includes a price for changing memory: the controller is judged by validity loss plus actuation cost.

The optimal scientific arc is now fixed:

> **law -> inference -> calibration -> control -> sequential validation**

That is the story to optimize from now on.
