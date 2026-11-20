# Critical analysis

All numbers below come from `results/summary.csv`, `results/paired_mc_vs_pi.csv` and
`results/key_numbers.json`: 61 scenarios × 20 replicates, read at the final iteration. "Better/worse
than" means the 95% Monte Carlo CI excludes the reference value.

## What the results say

**Cold-start probabilities are mostly worse than a constant predictor.** For the main
block (onlyX × A, MC estimator), relative KL is above 1 in **53 of 61** scenarios and below
1 in only **6**: all have α ∈ {−2, 0}, with n ≥ 700 and p ≤ 700 (Fig. 1). So in most of the
design, knowing only the block's edge density would beat the model's cold-start
predictions. Relative Frobenius error is above 1 in 43 of 61 scenarios. This value has a
direct interpretation: ‖P̂ − P‖² > ‖P‖² holds exactly when ‖P̂‖² > 2⟨P̂, P⟩. In those
scenarios, then, the estimate puts more probability mass on the wrong pairs than on
the right ones. It is not just under-predicting a sparse graph. The in-network block
behaves differently: A × A has relative KL < 1 in 53 of 61 scenarios, and the 8 exceptions
all have n = 100 or α = −8. The pipeline does learn the network. What fails is the
transfer of that geometry to nodes seen only through their features.

**Hypotheses.** H1 (error falls with n) holds, with diminishing returns after n ≈ 700.
H3 (sparser is harder) holds strongly. **H2 is reversed:** in every row of both
slices that vary p, error *increases* monotonically in p (Fig. 4: at n = 400, α = −6, relative KL
goes from 1.33 at p = 100 to 7.41 at p = 1500). **H4 is rejected:** the paired MC − plug-in difference favours MC in
45 scenarios, but it favours the plug-in in 14, all with α ∈ {−8, −6} and p ≤ 700 (Fig. 3).

**The training dynamics explain more than the final numbers do (Fig. 2).** The plug-in relative KL of the main
block reaches its minimum at iteration 49 in 46 scenarios and at iteration 99 in 12 more. The
median ratio of the final value to this minimum is 2.2. Meanwhile, A × A keeps improving
(it is lower at the end than at iteration 49 in 54 of 61 scenarios). At its best iteration, the plug-in cold-start
estimate has mean relative KL below 1 in 33 of 61 scenarios; at the final iteration only 4 do. Training to
convergence therefore optimises the observed blocks at the expense of the cold-start block. The posterior log-variances settle
between −6.0 and −2.2 (posterior SD 0.05–0.33, against a prior SD of 1), which is why MC and plug-in differ
by less than 0.1 in relative KL in 38 of 61 scenarios. They differ most where the plug-in error is largest
(n = 100, large p). Averaging sigmoids over posterior draws tempers extreme
probabilities (Jensen), and KL penalises confident errors heavily. In the sparse,
small-p cases the same tempering pushes correctly small probabilities upward, so the plug-in wins.

**Mechanisms I consider plausible (not tested separately).** (i) *Loss weighting:* the X term
sums n_X · p Bernoulli terms, while the A term sums about n_A²/2. Raising p shifts the objective
toward reconstructing features, so the latent geometry is shaped by X rather than by
inner products. This also explains the reversed p effect. (ii) *Encoder input shift:* network
nodes enter the shared weights through GCN-smoothed features ÂX, while onlyX nodes enter
through raw X. The two input distributions differ, and nothing in the loss ties onlyX
embeddings to the edge decoder. (iii) A related observation I cannot yet explain: onlyX × both
is worse than onlyX × onlyA in all 61 scenarios.

## Design choices and fairness

*Why these DGPs.* The latent-position model makes P known exactly. Its link function,
latent dimension and prior variance match the fitted model, so the edge model is
**well-specified on purpose**. This design favours the method, which makes the negative cold-start result harder to
dismiss as misspecification. X depends on Z through a random ReLU network, which is close to, but not
inside, the decoder family. The α grid spans 1%–50% density. α = −6 (3.7%, expected degree ≈ 26
at n = 700) is the realistic sparse anchor, and n = 100 is a deliberately data-poor stress case.
The three-slice design covers all two-way interactions without the cost of the full 125-cell cube.

*Fairness.* Only one method is fitted, so neutrality between competitors is not the issue here.
Fairness to the method under each condition is. Hyperparameters (lr, L2, 1,200 iterations, unit loss
weights) were held fixed rather than tuned per scenario. That keeps the comparison clean, but the
fixed X:A weighting is effectively a different trade-off at each p, so the "p effect"
is confounded with loss weighting. The constant-density baseline is
generous to the baseline, because it uses the *true* p̄. The relative logit error ‖Θ̂ − Θ‖/‖Θ‖ is not comparable
across α, because ‖Θ‖² contains a term proportional to α² that dominates at α = −8. For that reason I do not use it for conclusions.

## Limitations and scenarios not included

The study uses one node split (30/40/30), one latent dimension (d_fit = d_true = 8), and a
DGP with no degree heterogeneity, community structure or misspecified link, all of which matter for real networks.
It has no early stopping and no tuning, and it includes no external competitor, such as a features-only regression of
μ on X, or a GCN trained with link-prediction loss on held-out edges. Sparsity is fixed per scenario
instead of scaling with n, so the large-n results describe a dense-regime asymptotic.
The MC estimator was evaluated only at the final iteration.

## Implications and next steps

As configured, this VAE's cold-start link probabilities should not be used as calibrated
probabilities. A model selected by in-network fit (A × A) or by training loss can be
selected at exactly the wrong point for cold-start prediction. Reports for this model
should include a relative-to-constant baseline. Next, I would: (1) add early stopping on held-out
cold-start pseudo-nodes (mask the edges of some `both` nodes); (2) reweight the X likelihood
by 1/p and rerun the p slice to test mechanism (i); (3) add the features-only and
link-prediction GCN baselines; (4) vary the split proportions and misspecify d.

## Implementation challenges and confidence

The hardest parts were numerical. Edge probabilities near 0 underflow in float32 and make KL unstable, so KL and the MC average are
computed in log space (`logsigmoid`, running `logaddexp`). The other challenge was refactoring a working server
script into modules without changing a single number. `tests/test_simulation.py` runs the
original script and the refactored pipeline with the same seed and checks that they agree.

The Monte Carlo error is small: the median relative MCSE of the main relative KL is 1.5% (max 6.3%), and every
qualitative claim above holds with CIs far from the thresholds. Four things limit my confidence:
(a) the cached results came from the original script, which **did not seed torch**. Data and
splits are exactly reproducible, but a rerun gives slightly different training noise. The refactor now
seeds torch, and GPU kernels may still be nondeterministic. (b) Every conclusion is conditional on
one hyperparameter configuration, and Fig. 2 shows that the stopping time alone changes the answer by a factor of about 2.
(c) Common random numbers across scenarios make the CIs marginal rather than joint. (d) Each
replicate is a single training run, so optimisation variance is folded into replicate variance.

