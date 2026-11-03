# suim_nonwater, stride 3: mean IoU (dendrogram variants vs k-means baseline)

k-means baseline (k<=6, border rule / oracle at silhouette k*): rgb: 0.513 / 0.778, 768: 0.475 / 0.774

| feat | spatial | linkage | n | sil_border | gap_border | subtree | score | oracle_sil | oracle_gap |
|---|---|---|---|---|---|---|---|---|---|
| 768 | free | ward | 92 | 0.486 | 0.491 | 0.450 | 0.278 | 0.745 | 0.696 |
| 768 | free | average | 92 | 0.433 | 0.442 | 0.472 | 0.272 | 0.733 | 0.640 |
| 768 | free | complete | 92 | 0.465 | 0.489 | 0.452 | 0.252 | 0.711 | 0.658 |
| 768 | free | single | 92 | 0.508 | 0.521 | 0.487 | 0.228 | 0.521 | 0.521 |
| 768 | constrained | ward | 92 | 0.461 | 0.472 | 0.433 | 0.270 | 0.761 | 0.715 |
| 768 | constrained | average | 92 | 0.201 | 0.202 | 0.446 | 0.233 | 0.520 | 0.520 |
| 768 | constrained | complete | 92 | 0.252 | 0.376 | 0.399 | 0.206 | 0.517 | 0.526 |
| 768 | constrained | single | 92 | 0.283 | 0.236 | 0.467 | 0.236 | 0.520 | 0.520 |
| rgb | free | ward | 92 | 0.485 | 0.478 | 0.448 | 0.290 | 0.772 | 0.740 |
| rgb | free | average | 92 | 0.515 | 0.493 | 0.459 | 0.285 | 0.771 | 0.756 |
| rgb | free | complete | 92 | 0.488 | 0.475 | 0.465 | 0.310 | 0.761 | 0.760 |
| rgb | free | single | 92 | 0.128 | 0.168 | 0.493 | 0.320 | 0.520 | 0.520 |
| rgb | constrained | ward | 92 | 0.453 | 0.446 | 0.437 | 0.272 | 0.756 | 0.742 |
| rgb | constrained | average | 92 | 0.119 | 0.110 | 0.458 | 0.258 | 0.515 | 0.517 |
| rgb | constrained | complete | 92 | 0.308 | 0.323 | 0.428 | 0.235 | 0.494 | 0.513 |
| rgb | constrained | single | 92 | 0.044 | 0.043 | 0.473 | 0.246 | 0.520 | 0.520 |
