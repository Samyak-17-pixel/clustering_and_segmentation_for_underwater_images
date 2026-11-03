# suim_saliency, stride 14: mean IoU (dendrogram variants vs k-means baseline)

k-means baseline (k<=6, border rule / oracle at silhouette k*): rgb: 0.385 / 0.573, 768: 0.424 / 0.584

| feat | spatial | linkage | n | sil_border | gap_border | subtree | score | oracle_sil | oracle_gap |
|---|---|---|---|---|---|---|---|---|---|
| 768 | free | ward | 108 | 0.432 | 0.416 | 0.309 | 0.377 | 0.607 | 0.502 |
| 768 | free | average | 108 | 0.472 | 0.537 | 0.303 | 0.383 | 0.626 | 0.616 |
| 768 | free | complete | 108 | 0.450 | 0.445 | 0.304 | 0.384 | 0.593 | 0.546 |
| 768 | free | single | 108 | 0.190 | 0.192 | 0.261 | 0.332 | 0.203 | 0.200 |
| 768 | constrained | ward | 108 | 0.475 | 0.435 | 0.308 | 0.331 | 0.556 | 0.498 |
| 768 | constrained | average | 108 | 0.109 | 0.092 | 0.267 | 0.324 | 0.245 | 0.225 |
| 768 | constrained | complete | 108 | 0.283 | 0.261 | 0.265 | 0.295 | 0.366 | 0.314 |
| 768 | constrained | single | 108 | 0.051 | 0.059 | 0.246 | 0.263 | 0.199 | 0.200 |
| rgb | free | ward | 108 | 0.395 | 0.369 | 0.318 | 0.412 | 0.567 | 0.520 |
| rgb | free | average | 108 | 0.392 | 0.409 | 0.311 | 0.403 | 0.580 | 0.569 |
| rgb | free | complete | 108 | 0.355 | 0.345 | 0.301 | 0.400 | 0.561 | 0.558 |
| rgb | free | single | 108 | 0.055 | 0.058 | 0.291 | 0.424 | 0.216 | 0.204 |
| rgb | constrained | ward | 108 | 0.414 | 0.399 | 0.313 | 0.343 | 0.505 | 0.493 |
| rgb | constrained | average | 108 | 0.149 | 0.133 | 0.269 | 0.362 | 0.280 | 0.260 |
| rgb | constrained | complete | 108 | 0.234 | 0.222 | 0.268 | 0.351 | 0.322 | 0.318 |
| rgb | constrained | single | 108 | 0.026 | 0.036 | 0.253 | 0.347 | 0.199 | 0.200 |
