# PLAN

## Status

Este repositório já não é mais um conjunto de linhas paralelas sem relação clara.
O objeto central ficou nítido:

- **validade temporal sob drift**
- **horizonte prospectivo de validade**
- **inferência bootstrap-calibrada do horizonte**
- **controle operacional de memória**

O trabalho recente dissipou incertezas importantes:

- o horizonte não é só uma metáfora; ele é um objeto constitutivo com lei explícita
- a inferência via geometria de lag funciona sob especificação correta
- misspecification de forma é detectável por diagnósticos locais
- heterocedasticidade afeta principalmente cobertura e thresholds, não o ponto estimado
- o controller só se torna realmente discriminativo quando a memória vira um **estado controlado** com inércia e custo de update

O plano a partir de agora é **consolidar** tudo isso no paper raiz e matar os spin-offs como narrativas independentes. Eles passam a existir apenas como superfícies temporárias de trabalho e migração.

## Core Thesis

O paper raiz deve afirmar uma única tese forte:

> Sob drift, a persistência inferencial é governada por uma lei constitutiva de validade temporal que induz um horizonte ótimo de memória, permite inferência calibrada a partir da geometria de lag e fecha operacionalmente em uma política de controle de memória superior a detectores e políticas ingênuas.

Em termos formais, o objeto principal permanece:

`E d(\widehat P_t^{(n)}, P_t) <= C_K n^{-a} + C_S \zeta n^H`

com horizonte induzido

`n^*(a,H) ~ (C_K / \zeta)^{1/(a+H)}`.

Tudo o que entra no manuscrito final deve servir diretamente a essa tese.

## What We Now Know

### 1. The constitutive law is the real center

O melhor enquadramento do projeto não é drift detection, nem adaptation burden, nem package design.
O centro é:

- o **campo de validade** `V(\phi, \tau)`
- a decomposição entre termo finito-amostral e termo de staleness
- a geometria do horizonte e da banda de memória útil

### 2. Horizon inference works

No workspace `projects/scale-consistency`, a linha `horizon_bridge` mostrou:

- recovery forte sob especificação correta
- grids ampliados com `lag_count`, `n`, `H`, `\zeta`, `sigma0`
- misspecification stress test com `sinusoid`, `piecewise`, `mixed`, `bump`, `slope_shift`
- diagnósticos úteis: KL residual, KL standardized, curvature, Durbin-Watson, periodogram

O resultado importante aqui é: a geometria de lag realmente carrega informação operacional sobre `H`, `\zeta` e `n^*`.

### 3. Heteroskedasticity changes inference, not the point estimate

O trabalho com:

- `variance_bridge.py`
- sweeps hetero fortes (`power`, `jump`, `ar`)
- bootstrap `parametric`, `wild`, `moving_block`

fixou uma descoberta central:

- **modelagem explícita de variância não move materialmente `H` nem `n^*` nos regimes testados**
- **bootstrap robusto melhora cobertura e thresholds de forma substantiva**

Leitura científica:

- heterocedasticidade não exige reescrever o estimador pontual
- ela exige inferência mais honesta

### 4. The controller only separated once memory became a controlled state

O primeiro controller mínimo tinha boa lógica de decisão, mas não separava bem `controller` de `detector_only` na camada de loss.

O salto veio quando `n_t` virou:

- estado controlado
- com atualização amortecida
- deadband / histerese
- custo explícito de update

Essa foi a peça operacional faltante.

### 5. The sequential benchmark is now scientifically meaningful

No `projects/temporalbridge`, o benchmark sequencial com schedule temporal e memória dinâmica já mostra:

- `controller` com alta acurácia de ação
- lead time positivo em média
- falso-alarme muito baixo
- excess validity loss pequeno
- separação real contra `detector_only`, `deploy_only` e `fixed_policy`

Na forma atual do benchmark:

- `controller`: excess baixo, regret baixo
- `detector_only`: mantém lead time, mas paga muito mais em validity loss
- `deploy_only`: às vezes razoável em loss, mas colapsa em lead time
- `fixed_policy`: colapso completo

## Scientific Claims

### Closed / strong claims

1. Existe uma **lei constitutiva** do horizonte de validade temporal sob drift.
2. O horizonte `n^*` é imposto pela geometria do problema, não por tuning contingente.
3. `H`, `\zeta` e `n^*` são inferíveis a partir da geometria de lag sob o modelo correto.
4. Heterocedasticidade nos regimes testados não altera materialmente o ponto estimado de `H` e `n^*`.
5. Bootstrap robusto (`wild`, `moving_block`) melhora substancialmente cobertura e calibração de thresholds.
6. Um controller bootstrap-calibrado com memória dinâmica supera baselines simples em benchmark sequencial sintético.

### Negative results that matter

1. `variance_bridge` não justificou promoção da modelagem explícita de variância ao núcleo do estimador.
2. Detectores sozinhos não resolvem o problema operacional; é a política de memória que importa.
3. No outro eixo do repo, a claim forte de suficiência pre-drift para PHH não se sustentou em dados reais. Isso é útil porque impede que a linha PHH capture a narrativa principal por inércia.

### Open claims / still frontier

1. bound formal de regret do controller
2. integração teórica plena de `\tau_valid`, `\tau_detect`, `\Delta_{val-det}`
3. replay convincente em streams reais
4. teoria mais apertada para misspecification misto forma+escala

## Why This Is Scientifically Strong

O projeto hoje entrega algo raro:

- uma **lei constitutiva**
- uma **inferência calibrada** dessa lei
- um **fecho operacional** em controle de memória

Isso é muito mais forte do que:

- mais um detector de drift
- mais um benchmark empírico de adaptation
- mais uma heurística de update policy

O resultado final é um novo objeto de ML/streaming inference:

- **validade temporal prospectiva**
- com **horizonte estimável**
- e **alarmes / decisões calibrados**

## Information Architecture

### Root paper

O `main.tex` na raiz deve ser o único manuscrito ativo.

Ele deve absorver:

- a teoria do horizonte
- a inferência por geometria de lag
- a calibração bootstrap
- o controller sequencial

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

### Arc 1. Constitutive law

- definição de `V(\phi,\tau)`
- lei `C_K n^{-a} + C_S \zeta n^H`
- geometria de `n^*`
- benchmark Gaussiano

### Arc 2. Inference and calibration

- estimadores de `H`, `\zeta`, `n^*`
- CLT / delta method / identifiability
- recovery sob especificação correta
- misspecification diagnostics
- bootstrap robusto

### Arc 3. Operational closure

- memória como estado controlado
- controller
- benchmark sequencial
- validity loss, excess loss, regret, lead time

Essa é a narrativa. Tudo o que não servir diretamente a esse arco deve ir para apêndice, nota de projeto, ou ser congelado.

## Root Paper Structure

1. **Introduction**
2. **Constitutive Horizon Law**
3. **Horizon Inference from Lag Geometry**
4. **Bootstrap Calibration and Misspecification**
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
- sequential benchmark with full validity curve and memory dynamics
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
   - rewrite framing around one object

2. **Controller ablations**
   - `wild` vs `moving_block`
   - update cost low/high
   - tracking gain low/high
   - deadband / persistence

3. **Integrate the delay object explicitly**
   - make `\tau_valid`, `\tau_detect`, `\Delta_{val-det}` first-class in the sequential benchmark and writeup

4. **Real-stream replay**
   - ELEC2 / MOA-like streams first
   - then only harder domains if needed

5. **Theory sharpening**
   - tighter lower theory beyond the benchmark sign law
   - regret-style or validity-loss guarantees for the controller

## Limitations

1. Sequential theory is still weaker than the empirical controller results.
2. Real-stream evidence for the controller is not yet in place.
3. The current sequential schedule is informative, but still synthetic and hand-built.
4. Some project branches in the repo still reflect exploratory phases that should no longer dictate the main story.
5. `temporalbridge` still depends on code in `projects/scale-consistency`; migration is not complete.

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

4. Integrate `\Delta_{val-det}` into the sequential benchmark outputs.

5. Run short real-stream replay before any final paper freeze.

## Bottom Line

The project is now coherent.

We are no longer trying to understand what the contribution is.
We now know it:

> a constitutive law of temporal validity under drift, an inferential pipeline that recovers the induced horizon with honest uncertainty, and a memory controller that uses that horizon to make better sequential decisions than detector-only or naive baselines.

That is the story to optimize from now on.
