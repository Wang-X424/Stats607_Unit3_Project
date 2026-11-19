# Stability report

## Cached Unit 2 results (61 scenarios, 1220 replicates)

* replicates with any non-finite logged value: **3** (0.2%); scenarios: N100_X700_Aneg8
* non-finite keys: mc_relfro_avg_onlyX_both, mc_relfro_avg_onlyX_onlyA, pi_relfro_avg_onlyX_both, pi_relfro_avg_onlyX_onlyA
* final loss finite in all replicates: True
* relative loss decrease over the last 100 iterations: median 1.14e-03, max 5.36e-02, replicates where the loss went UP: 376
* mean posterior log-variance range over all logs: [-6.23, 0.78]

## DGP headroom (20 replicates x 61 scenarios; graph checks on 3)

* replicates where the Unit 2 naive sigmoid overflowed float32 exp (RuntimeWarning): **262 / 1220** (21.5%) in 61 scenarios
* most negative float32 logit: A -31.5, X -144.9 (exp overflows float32 beyond 88.7); smallest headroom -56.2
* largest logit: A 36.5, X 131.9
* observed graphs with zero edges: 0
* isolated observed nodes: max fraction 0.63 (at N100_X700_Aneg8)
* onlyA nodes with no 'both' neighbour (global-mean fill): max fraction 0.73 (at N100_X700_Aneg8)

## Live training with warnings as errors

```
  n    p  alpha  rep error  all_losses_finite  max_abs_logvar  lv_bound
100 1500     -8    0                     True        6.898096      30.0
100 1500     -8    1                     True        8.099445      30.0
100 1500      0    0                     True        8.326451      30.0
100 1500      0    1                     True        8.946146      30.0
100  100     -8    0                     True        3.467803      30.0
100  100     -8    1                     True        3.826391      30.0
100  100      0    0                     True        3.890924      30.0
100  100      0    1                     True        4.358397      30.0
400 1500     -8    0                     True       11.887848      30.0
400 1500     -8    1                     True       10.625340      30.0
```
