# suim_saliency, stride 3: mean IoU (dendrogram variants vs k-means baseline)

k-means baseline (k<=6, border rule / oracle at silhouette k*): rgb: 0.375 / 0.575, 768: 0.429 / 0.595

| feat | spatial | linkage | n | sil_border | gap_border | subtree | score | oracle_sil | oracle_gap |
|---|---|---|---|---|---|---|---|---|---|
| 768 | free | ward | 108 | 0.475 | 0.417 | 0.313 | 0.398 | 0.612 | 0.506 |
| 768 | free | average | 108 | 0.519 | 0.552 | 0.314 | 0.395 | 0.642 | 0.628 |
| 768 | free | complete | 108 | 0.448 | 0.428 | 0.298 | 0.383 | 0.577 | 0.537 |
| 768 | free | single | 108 | 0.198 | 0.200 | 0.256 | 0.299 | 0.200 | 0.200 |
| 768 | constrained | ward | 108 | 0.483 | 0.435 | 0.315 | 0.347 | 0.564 | 0.501 |
| 768 | constrained | average | 108 | 0.059 | 0.057 | 0.249 | 0.257 | 0.199 | 0.200 |
| 768 | constrained | complete | 108 | 0.129 | 0.156 | 0.234 | 0.226 | 0.219 | 0.214 |
| 768 | constrained | single | 108 | 0.096 | 0.084 | 0.234 | 0.275 | 0.199 | 0.199 |
| rgb | free | ward | 108 | 0.402 | 0.402 | 0.318 | 0.407 | 0.591 | 0.543 |
| rgb | free | average | 108 | 0.378 | 0.398 | 0.316 | 0.413 | 0.570 | 0.572 |
| rgb | free | complete | 108 | 0.360 | 0.352 | 0.312 | 0.434 | 0.557 | 0.558 |
| rgb | free | single | 108 | 0.052 | 0.072 | 0.296 | 0.409 | 0.199 | 0.199 |
| rgb | constrained | ward | 108 | 0.431 | 0.422 | 0.319 | 0.372 | 0.520 | 0.519 |
| rgb | constrained | average | 108 | 0.062 | 0.050 | 0.259 | 0.347 | 0.215 | 0.206 |
| rgb | constrained | complete | 108 | 0.165 | 0.172 | 0.254 | 0.316 | 0.257 | 0.260 |
| rgb | constrained | single | 108 | 0.025 | 0.023 | 0.251 | 0.352 | 0.199 | 0.199 |
