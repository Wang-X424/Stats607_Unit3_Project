# ADEMP: cold-start edge-probability recovery with a graph VAE

**Option B (original simulation study).** The study asks how well a variational
autoencoder with a GCN encoder recovers the *edge-probability matrix* for nodes whose
network connections are completely unobserved ("cold-start" nodes) and only have
node features. The DGP is a latent-position network model, so the true probabilities
are known and the estimate can be compared with them directly, without going through
sampled edges (AUC).

---

## Aims

1. **Primary.** Quantify how accurately the model recovers the true edge probabilities
   between feature-only nodes (`onlyX`) and network-observed nodes (`A` = `onlyA` ∪ `both`),
   and how this accuracy depends on
   * the number of nodes *n*,
   * the feature dimension *p*,
   * the baseline sparsity α (graph density).
2. **Benchmark against a trivial predictor.** Check whether the estimate beats the best
   constant predictor, i.e. a predictor that knows only the true density p̄ of the block.
   Relative KL > 1 means "worse than knowing the density".
3. **Estimator comparison.** Compare the plug-in estimator σ(μᵢᵀμⱼ + b) with the Monte Carlo
   posterior-mean estimator E_q[σ(zᵢᵀzⱼ + b)].
4. **Secondary.** Compare the cold-start block with blocks whose nodes are observed in
   the network (A × A), and track how accuracy changes during training.

Working hypotheses before running: (H1) error decreases in *n*; (H2) error decreases in *p*,
because more features carry more information about Zᵢ; (H3) sparser graphs (smaller α) are
harder; (H4) MC is never worse than plug-in because it propagates posterior uncertainty.

## Data-generating mechanism

For each replicate *r* = 0, …, 19 with seed *s* = 42 + *r*:

| Step | Definition |
|---|---|
| Latent positions | Zᵢ ~ N(0, I₈) i.i.d., *i* = 1..*n* (d = 8, Var(Z_ik) = 1); `np.random.RandomState(s)` |
| True logits / estimand | Θᵢⱼ = α + ZᵢᵀZⱼ,  Pᵢⱼ = σ(Θᵢⱼ) |
| Network | Aᵢⱼ ~ Bernoulli(Pᵢⱼ) for *i* < *j*, symmetric, zero diagonal (same RNG stream) |
| Features | Xᵢₖ ~ Bernoulli(σ([W₂ ReLU(W₁Zᵢ + b₁) + b₂]ₖ)), *k* = 1..*p*; W₁ ∈ ℝ^{8×32}, W₂ ∈ ℝ^{32×p} with N(0,1) entries, b ~ N(0, 0.1²); `RandomState(s + 1000)` |
| Node types | random permutation (`RandomState(s + 100)`): 30% `onlyA` (network only), 40% `both`, 30% `onlyX` (features only, **no edges observed**) |
| Observed data | A restricted to `onlyA ∪ both`; X restricted to `both ∪ onlyX` |

Average edge density E[Pᵢⱼ] by α (computed by Monte Carlo over Z): α = −8: 1.1%,
−6: 3.7%, −4: 10.9%, −2: 26.4%, 0: 50%.

### Factors varied

| Factor | Levels | Fixed value in the other slices |
|---|---|---|
| *n* (nodes) | 100, 400, 700, 1100, 1500 | 700 |
| *p* (feature dim.) | 100, 400, 700, 1100, 1500 | 700 |
| α (sparsity) | −8, −6, −4, −2, 0 | −6 |

Held fixed: d = 8, Var(Z) = 1, node split 30/40/30, 20 replicates per scenario.

### Design matrix

The design is the union of three full two-way factorials (slices), with one factor held at
its fixed value in each slice. After removing duplicates there are **61 unique scenarios**
(25 + 25 + 25 − 14 overlaps), each with 20 replicates = 1,220 fitted models.
`results/design.csv` lists all 61 with node counts.

**Slice 1: *n* × *p* at α = −6 (25 scenarios)**

| *n* \ *p* | 100 | 400 | 700 | 1100 | 1500 |
|---|:-:|:-:|:-:|:-:|:-:|
| 100  | ✓ | ✓ | ✓ | ✓ | ✓ |
| 400  | ✓ | ✓ | ✓ | ✓ | ✓ |
| 700  | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1100 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1500 | ✓ | ✓ | ✓ | ✓ | ✓ |

**Slice 2: *p* × α at *n* = 700 (25 scenarios; the α = −6 column repeats slice 1)**

| *p* \ α | −8 | −6 | −4 | −2 | 0 |
|---|:-:|:-:|:-:|:-:|:-:|
| 100  | ✓ | (S1) | ✓ | ✓ | ✓ |
| 400  | ✓ | (S1) | ✓ | ✓ | ✓ |
| 700  | ✓ | (S1) | ✓ | ✓ | ✓ |
| 1100 | ✓ | (S1) | ✓ | ✓ | ✓ |
| 1500 | ✓ | (S1) | ✓ | ✓ | ✓ |

**Slice 3: *n* × α at *p* = 700 (25 scenarios; the α = −6 column repeats S1, the *n* = 700 row repeats S2)**

| *n* \ α | −8 | −6 | −4 | −2 | 0 |
|---|:-:|:-:|:-:|:-:|:-:|
| 100  | ✓ | (S1) | ✓ | ✓ | ✓ |
| 400  | ✓ | (S1) | ✓ | ✓ | ✓ |
| 700  | (S2) | (S1) | (S2) | (S2) | (S2) |
| 1100 | ✓ | (S1) | ✓ | ✓ | ✓ |
| 1500 | ✓ | (S1) | ✓ | ✓ | ✓ |

New scenarios: 25 + 20 + 16 = 61.

## Estimands / targets

The estimand is the true edge-probability matrix **P** = σ(α + ZZᵀ), evaluated block by block
(rows × columns in training order):

| Block | Rows × columns | Role |
|---|---|---|
| `onlyX_A` | onlyX × (onlyA ∪ both) | **main target: cold-start link probabilities** |
| `onlyX_onlyA` | onlyX × onlyA | sub-block of the main target |
| `onlyX_both` | onlyX × both | sub-block of the main target |
| `A_A` | (onlyA ∪ both)², diagonal excluded | in-network reference block |

Θ is identified only up to the model class (Z only up to rotation), so the targets are P
and Θ, which are rotation-invariant, and not Z itself.

## Methods

**Model.** VAE with latent dimension 8 (= truth) and hidden width 128.
* Encoder q(zᵢ | ·) = N(μᵢ, diag σᵢ²): for network-observed nodes, a two-layer GCN on
  Â = D̃^{-1/2}(A + diag(deg))D̃^{-1/2}, applied to features in which the missing X of
  `onlyA` nodes are filled with the average X of their `both` neighbours (or with the global
  mean if they have none); for `onlyX` nodes, the same weights applied as a plain MLP to their raw X.
* Decoders: P(Aᵢⱼ = 1 | z) = σ(zᵢᵀzⱼ + b), with b initialised at the observed logit
  density; X decoder = one-hidden-layer ReLU network.
* Loss = summed BCE over observed edges (upper triangle) + summed BCE over observed
  features + KL(q ‖ N(0, τ²I)) with τ² = 1 (= truth) + 0.01 × ½‖W‖² (L2 on weight matrices).
* Optimiser: Adam, lr 1e-3, cosine annealing to 1e-5 over 1,200 full-batch iterations;
  one reparameterised sample per iteration. No early stopping.

**Estimators of P compared** (both computed from the same fitted model):
* **plug-in (`pi`)**: P̂ᵢⱼ = σ(μᵢᵀμⱼ + b);
* **MC posterior mean (`mc`)**: P̂ᵢⱼ = (1/S)Σₛ σ(zᵢ⁽ˢ⁾ᵀzⱼ⁽ˢ⁾ + b), zᵢ⁽ˢ⁾ ~ q, S = 50,
  accumulated in log space with logsumexp.

**Reference predictor (implicit):** the constant p̄ = mean of P over the block. It is the
denominator of relative KL, so the comparison with it needs no extra fitting.

## Performance measures

Computed at the final iteration (1199) for each replicate, in float64:

| Measure | Definition | Reference value |
|---|---|---|
| **relative KL** (primary) | mean KL(Bern(Pᵢⱼ)‖Bern(P̂ᵢⱼ)) / mean KL(Bern(Pᵢⱼ)‖Bern(p̄)) | 0 perfect; **1 = constant-density predictor** |
| relative Frobenius (all) | ‖P̂ − P‖_F / ‖P‖_F | 1 = predicting P̂ ≡ 0 |
| relative Frobenius (row avg.) | mean over rows with expected degree ≥ 1 of ‖P̂ᵢ − Pᵢ‖/‖Pᵢ‖ | 1 = zero predictor |
| centred rel. Frobenius | ‖P̂ − P‖_F / ‖P − p̄‖_F (1 − value² is an R²) | 1 = constant predictor |
| RMSE, KL | unnormalised versions | — |
| logit-scale error (plug-in) | ‖Θ̂ − Θ‖_F / ‖Θ‖_F | — |

Summaries across the R = 20 replicates: mean, SD, **Monte Carlo standard error**
SD/√R, and 95% CI mean ± 1.96·MCSE. The MC − plug-in comparison uses **paired**
differences within replicate. Diagnostics: the plug-in metrics and the mean posterior
log-variance are also logged at iterations 0, 49, 99, …, 1199 (MC only at the final
iteration, for cost).

## Reproducing

All values above are in `src/config.py`. `make all` reruns any scenario that is not in
`results/raw/` (the shipped cache holds all 61) and regenerates the tables and figures.
The data and split for replicate *r* depend only on seed 42 + *r*. The same seeds are
used in every scenario, so scenarios share common random numbers (for example, the first
rows of Z are identical across *n*). See README → *Reproducibility* for how torch is seeded.
