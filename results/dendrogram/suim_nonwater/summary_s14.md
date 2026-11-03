# suim_nonwater, stride 14: mean IoU (dendrogram variants vs k-means baseline)

k-means baseline (k<=6, border rule / oracle at silhouette k*): rgb: 0.505 / 0.776, 768: 0.491 / 0.766

| feat | spatial | linkage | n | sil_border | gap_border | subtree | score | oracle_sil | oracle_gap |
|---|---|---|---|---|---|---|---|---|---|
| 768 | free | ward | 92 | 0.505 | 0.524 | 0.452 | 0.274 | 0.747 | 0.690 |
| 768 | free | average | 92 | 0.458 | 0.467 | 0.473 | 0.273 | 0.733 | 0.655 |
| 768 | free | complete | 92 | 0.495 | 0.492 | 0.472 | 0.266 | 0.720 | 0.674 |
| 768 | free | single | 92 | 0.480 | 0.492 | 0.500 | 0.249 | 0.521 | 0.521 |
| 768 | constrained | ward | 92 | 0.467 | 0.462 | 0.432 | 0.280 | 0.746 | 0.722 |
| 768 | constrained | average | 92 | 0.128 | 0.132 | 0.452 | 0.256 | 0.535 | 0.524 |
| 768 | constrained | complete | 92 | 0.268 | 0.310 | 0.426 | 0.233 | 0.527 | 0.532 |
| 768 | constrained | single | 92 | 0.097 | 0.121 | 0.462 | 0.236 | 0.520 | 0.520 |
| rgb | free | ward | 92 | 0.537 | 0.514 | 0.477 | 0.322 | 0.767 | 0.723 |
| rgb | free | average | 92 | 0.534 | 0.548 | 0.478 | 0.322 | 0.770 | 0.764 |
| rgb | free | complete | 92 | 0.511 | 0.508 | 0.472 | 0.303 | 0.767 | 0.763 |
| rgb | free | single | 92 | 0.103 | 0.135 | 0.512 | 0.321 | 0.533 | 0.519 |
| rgb | constrained | ward | 92 | 0.453 | 0.456 | 0.449 | 0.256 | 0.746 | 0.728 |
| rgb | constrained | average | 92 | 0.186 | 0.175 | 0.471 | 0.278 | 0.515 | 0.521 |
| rgb | constrained | complete | 92 | 0.300 | 0.275 | 0.453 | 0.279 | 0.521 | 0.525 |
| rgb | constrained | single | 92 | 0.058 | 0.102 | 0.478 | 0.262 | 0.520 | 0.520 |
