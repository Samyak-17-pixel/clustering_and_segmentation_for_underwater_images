# usod, stride 3: mean IoU (dendrogram variants vs k-means baseline)

k-means baseline (k<=6, border rule / oracle at silhouette k*): rgb: 0.467 / 0.671, 768: 0.523 / 0.702

| feat | spatial | linkage | n | sil_border | gap_border | subtree | score | oracle_sil | oracle_gap |
|---|---|---|---|---|---|---|---|---|---|
| 768 | free | ward | 300 | 0.571 | 0.544 | 0.390 | 0.494 | 0.716 | 0.635 |
| 768 | free | average | 300 | 0.630 | 0.646 | 0.382 | 0.499 | 0.711 | 0.696 |
| 768 | free | complete | 300 | 0.568 | 0.527 | 0.373 | 0.497 | 0.676 | 0.614 |
| 768 | free | single | 300 | 0.211 | 0.224 | 0.322 | 0.387 | 0.224 | 0.224 |
| 768 | constrained | ward | 300 | 0.603 | 0.577 | 0.389 | 0.467 | 0.705 | 0.649 |
| 768 | constrained | average | 300 | 0.072 | 0.085 | 0.309 | 0.358 | 0.224 | 0.224 |
| 768 | constrained | complete | 300 | 0.178 | 0.194 | 0.290 | 0.308 | 0.256 | 0.244 |
| 768 | constrained | single | 300 | 0.093 | 0.090 | 0.282 | 0.371 | 0.224 | 0.224 |
| rgb | free | ward | 300 | 0.491 | 0.504 | 0.392 | 0.548 | 0.680 | 0.653 |
| rgb | free | average | 300 | 0.500 | 0.538 | 0.391 | 0.546 | 0.675 | 0.670 |
| rgb | free | complete | 300 | 0.450 | 0.437 | 0.380 | 0.545 | 0.653 | 0.650 |
| rgb | free | single | 300 | 0.055 | 0.057 | 0.361 | 0.540 | 0.228 | 0.224 |
| rgb | constrained | ward | 300 | 0.576 | 0.566 | 0.392 | 0.492 | 0.661 | 0.653 |
| rgb | constrained | average | 300 | 0.075 | 0.050 | 0.322 | 0.455 | 0.244 | 0.228 |
| rgb | constrained | complete | 300 | 0.202 | 0.211 | 0.305 | 0.424 | 0.288 | 0.296 |
| rgb | constrained | single | 300 | 0.022 | 0.022 | 0.298 | 0.463 | 0.224 | 0.224 |
