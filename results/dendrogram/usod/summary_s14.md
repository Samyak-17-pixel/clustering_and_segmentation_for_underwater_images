# usod, stride 14: mean IoU (dendrogram variants vs k-means baseline)

k-means baseline (k<=6, border rule / oracle at silhouette k*): rgb: 0.448 / 0.668, 768: 0.531 / 0.703

| feat | spatial | linkage | n | sil_border | gap_border | subtree | score | oracle_sil | oracle_gap |
|---|---|---|---|---|---|---|---|---|---|
| 768 | free | ward | 300 | 0.560 | 0.541 | 0.382 | 0.513 | 0.712 | 0.628 |
| 768 | free | average | 300 | 0.625 | 0.646 | 0.375 | 0.503 | 0.709 | 0.691 |
| 768 | free | complete | 300 | 0.572 | 0.540 | 0.366 | 0.488 | 0.676 | 0.632 |
| 768 | free | single | 300 | 0.202 | 0.219 | 0.313 | 0.436 | 0.225 | 0.225 |
| 768 | constrained | ward | 300 | 0.596 | 0.557 | 0.384 | 0.449 | 0.692 | 0.631 |
| 768 | constrained | average | 300 | 0.167 | 0.147 | 0.342 | 0.437 | 0.327 | 0.301 |
| 768 | constrained | complete | 300 | 0.346 | 0.330 | 0.334 | 0.408 | 0.420 | 0.391 |
| 768 | constrained | single | 300 | 0.049 | 0.057 | 0.314 | 0.397 | 0.224 | 0.224 |
| rgb | free | ward | 300 | 0.471 | 0.485 | 0.389 | 0.539 | 0.676 | 0.645 |
| rgb | free | average | 300 | 0.490 | 0.527 | 0.386 | 0.544 | 0.679 | 0.673 |
| rgb | free | complete | 300 | 0.438 | 0.425 | 0.372 | 0.538 | 0.661 | 0.648 |
| rgb | free | single | 300 | 0.073 | 0.070 | 0.356 | 0.546 | 0.256 | 0.240 |
| rgb | constrained | ward | 300 | 0.552 | 0.545 | 0.381 | 0.480 | 0.646 | 0.632 |
| rgb | constrained | average | 300 | 0.194 | 0.138 | 0.343 | 0.478 | 0.335 | 0.285 |
| rgb | constrained | complete | 300 | 0.331 | 0.328 | 0.334 | 0.463 | 0.391 | 0.393 |
| rgb | constrained | single | 300 | 0.030 | 0.046 | 0.313 | 0.471 | 0.226 | 0.224 |
