# TODO: what we do next, and why

This is the working to-do list for the project. It turns the findings so far (README Sec. 10.8) into concrete
tasks. Every task says **what** it is, **why** we do it (the evidence behind it), **how** to implement it, **what it is
for**, and **when it counts as done**. `ROADMAP.md` has the same ideas grouped by phase; this file is the
actionable checklist.

The file has three parts:

* **Part A: committed next steps.** We will implement these, in this order.
* **Part B: recommended extras.** Worth doing once Part A is done, or if a Part A result points that way.
* **Part C: things we deliberately stop doing**, and why.

---

## 0. Where we stand (why this list looks the way it does)

Setting: frozen DINOv2 ViT-B/14, k-means on patch features, USOD (300 images). Numbers are for 768-D features, stride 3, k up to 6.

| Fact | Number | What it means for the plan |
|---|---|---|
| Oracle IoU (foreground chosen using the ground truth) | **0.702** | The clusters usually contain the object |
| Border-rule IoU (foreground chosen without labels) | **0.523** | What we can actually deploy today |
| Gap lost to choosing the foreground | **0.179** (significant, p < 0.001) | **Problem 1: the selection step** |
| Selection failures (clusters right, rule wrong) | **98 / 300** | Main target of Part A |
| Border rule when silhouette picks k\*=2 | 0.748 (oracle 0.759) | The rule works for 2 clusters... |
| Border rule when k\* = 3 to 6 | 0.36 to 0.49 | ...and breaks when it must merge 3 or more |
| Cluster failures (even the oracle is below 0.5) | **34 / 300**, 30 of them objects under 10% of the image | **Problem 2: small objects / feature resolution** |
| PCA-RGB background in the worst images | Regular grid/checkerboard texture | **Problem 3: likely DINOv2 feature artifacts** (registers fix these) |
| Stride 3 vs stride 14 (768-D) | No significant difference, 17x slower backbone | Stop using stride 3 |
| Speed of the deployable path | 2.5 to 6.5 FPS on a desktop GPU; CPU clustering dominates | **Problem 4: real time** |
| Empty-water frames | Never tested | **Problem 5: false alarms are unknown** |

So the order of work is: make experiments cheap (A1-A3), fix selection (A4), fix features (A5), measure what light
supervision gives (A6), and measure what the AUV actually needs (A7).

Legend for each task: **GPU** = needs a CUDA machine, **CPU** = can run here. Effort is a rough estimate.

---

## Part A: committed next steps (in order)

### A0. Housekeeping and reproducibility (CPU, about 1 hour)

- [ ] **A0.1 Save the border-rule masks, not only the oracle masks.**
  * **What:** in `src/dinov2_cluster_union.py` and `src/dinov2_cluster_union_stride.py`, also save the border-rule prediction to
    `masks_rgb_border/` and `masks_768_border/`.
  * **Why:** today `masks_rgb/` and `masks_768/` hold the **oracle** masks, which were chosen using the ground truth. Anyone
    who uses them as "predictions" gets numbers that are too optimistic.
  * **How:** in `process_feature`, keep the border-rule `prob` in the returned dict next to `border=metrics(...)`, and in
    `main` write it with the same `Image.fromarray(((prob > 0.5) * 255)...)` line used for the oracle mask.
  * **Done when:** both folders exist after a `--limit 10` run, and the IoU computed from the saved border masks matches `border_iou` in the CSV.
- [ ] **A0.2 Add `requirements.txt`** pinned to the versions in README Sec. 16 (torch, torchvision, numpy, scikit-learn,
  scipy, pillow, matplotlib, tqdm, pandas).
  * **Why:** this machine has no `.venv`, and the README's commands assume one. The GPU machine needs a one-line setup.
- [ ] **A0.3 Commit the current state** (analysis script, figures, README v1.1, roadmap, this file) before starting new experiments,
  so every new result can be compared against a fixed baseline commit.

### A1. Feature cache: extract DINOv2 features once (GPU, about half a day)

* **What:** a new script `extract_features.py` that runs the backbone once per image and stores everything later steps need.
* **Why:** the backbone takes about 25 ms per image, but every experiment so far re-ran it, then spent seconds in
  scikit-learn. Phases A4 to A6 are all "different processing of the same features". With a cache, each new selector or
  probe runs in minutes, even on a CPU, and results are exactly comparable because the features are identical.
* **For:** speed of iteration, and fairness of comparisons.
* **How:**
  1. Arguments: `--model` (`dinov2_vitb14`, `dinov2_vitb14_reg`, later S/L/g), `--height/--width` (448x588 and 644x868), `--out features/`.
  2. For each image save `features/<model>_<H>x<W>/<image>.npz` (fp16) with:
     * `patch`: `forward_features(x)["x_norm_patchtokens"]` reshaped to (gh, gw, D) (what we use today),
     * `cls`: `x_norm_clstoken` (for image-level "is there an object?" later),
     * `attn_cls`: last-block attention from the CLS token to every patch, per head (needed for A4.3),
     * `keys`: last-block key vectors per patch (needed for TokenCut in A4.4).
  3. Getting attention and keys: DINOv2 has no `get_last_selfattention`, so register a forward hook on
     `model.blocks[-1].attn.qkv` and rebuild them from its output:

     ```python
     qkv = out.reshape(B, N, 3, heads, D // heads).permute(2, 0, 3, 1, 4)   # 3, B, heads, N, hd
     q, k = qkv[0], qkv[1]
     attn = (q @ k.transpose(-2, -1) * (D // heads) ** -0.5).softmax(-1)  # B, heads, N, N
     first_patch = 1 + model.num_register_tokens                           # 1 for plain, 5 for _reg models
     attn_cls = attn[0, :, 0, first_patch:]                                 # heads, gh*gw
     keys = k[0, :, first_patch:].permute(1, 0, 2).reshape(gh * gw, -1)    # gh*gw, D
     ```

  4. Add a `--features DIR` option to the clustering scripts that loads `patch` from the cache instead of running the model.
* **Done when:** re-running E4 (stride 14, k up to 6) from the cache reproduces `cluster_union_stride14_k6/metrics_union.csv`
  within 0.005 mean IoU. Disk use is about 1.5 GB per model and resolution.

### A2. GPU clustering instead of scikit-learn (GPU, about half a day)

* **What:** torch versions of PCA, k-means and silhouette.
* **Why:** README Sec. 12 shows the 768-D path spends about 360 ms per frame in scikit-learn k-means, while the backbone takes 25 ms.
  That is the main reason we are below the 5 to 10 FPS target. It also makes the A4/A5 sweeps faster.
* **For:** speed now (experiments) and later (real time on the vehicle).
* **How:** `torch.pca_lowrank(X, q=3)` for PCA; a batched Lloyd's k-means with k-means++ init and `n_init` restarts on the GPU;
  silhouette on a fixed 3,000-point sample using `torch.cdist`. Put them in a small module `gpu_cluster.py`.
* **Done when:** same mean IoU as the scikit-learn version within 0.01 on all 300 images, and per-image clustering time is reported.

### A3. Evaluation protocol we can trust (CPU, about 2 hours)

* **What:** fixed data splits, multiple seeds, and the ability to add new runs to `src/analyze_results.py`.
* **Why:** A4 tunes a few weights and A6 trains a probe. Without held-out folds, those numbers would be measured on the data
  they were tuned on. Seeds are needed because several current differences are only about 0.01 to 0.03 IoU.
* **For:** being able to say "method X is better" with evidence.
* **How:**
  1. `splits.json`: 5 folds of 60 images (seed 0). Anything trained or tuned is reported as the mean over held-out folds.
  2. Run the clustering with seeds 0 to 4; report the mean and standard deviation of the mean IoU.
  3. Add `--extra-run NAME=path/to/metrics_union.csv` to `src/analyze_results.py` so new runs appear in all figures and in
     the significance table, compared against the current E4/E3 baselines.
* **Done when:** `src/analyze_results.py --extra-run ...` produces the comparison figures and paired tests for any new run.

### A4. Better label-free foreground selection (GPU for the cache, then CPU; about 2 days). Highest priority.

* **What:** replace or improve the border rule with methods that do not need labels.
* **Why:** this is where most of the IoU is lost: 0.179 IoU and 98 images. The analysis pins it down further: the rule is
  fine for 2 clusters and fails when it must merge 3 to 6 (k\*=6 alone accounts for 45 of the 98 failures). The current
  multi-cluster rule ("every cluster whose border fraction is below average is foreground") marks reef and sea-floor
  clusters as foreground too.
* **For:** the realistic (deployable) score. Target: **label-free IoU at least 0.60** (from 0.52).
* **How:** one script `select_foreground.py --method ...` that reads the cache, runs the clustering, applies a selector and writes
  `results/usod/select_<method>/metrics_union.csv` in the existing CSV format. Methods, from cheapest to most involved:

- [ ] **A4.1 k=2 fallback.**
  * **Approach:** keep k\* for the oracle, but always select the foreground from the k=2 clustering.
  * **Reason to try it:** it costs nothing, and it directly tests the finding that the rule is good at k=2.
  * **Caveat:** E1 k=2 scored 0.490 overall, so this mainly serves as a lower bound to beat.
- [ ] **A4.2 Background prototype.**
  * **Approach:**
    1. Take the mean feature of the patches on the border ring, or 2 to 3 k-medoids of them if the background has several looks (water plus sea floor).
    2. Score each patch as `1 - max cosine similarity to the prototypes`.
    3. Either threshold that score with Otsu, or mark a cluster as foreground if its mean score is above the image mean.
  * **Reason to try it:** it keeps the border assumption, which works on average, but compares feature similarity instead of
    counting border pixels. A reef cluster that touches the border is then correctly "background-like" even if it is large.
- [ ] **A4.3 Attention saliency.**
  * **Approach:** average the CLS-to-patch attention over the heads, using `attn_cls` from the cache. Then pick the clusters
    whose mean attention is above the image mean.
  * **Reason to try it:** the CLS token learns to attend to the main object. This is the original DINO finding: the attention
    maps segment objects without labels. It is a different signal from the border rule, so it can fix the cases the border rule gets wrong.
  * **Note:** use `dinov2_vitb14_reg`. Without registers, DINOv2 attention has high-norm artifact tokens in the background.
- [ ] **A4.4 TokenCut (spectral bipartition).**
  * **Approach:**
    1. Build a graph over patches from the last-layer `keys`: edge weight 1 if cosine ≥ 0.2, else 1e-5.
    2. Solve `(D - W) x = λ D x` and take the second-smallest eigenvector.
    3. Split the patches at its mean.
    4. Foreground is the side that contains the largest absolute eigenvector value.
  * **Reason to try it:** it is a well-known training-free method for unsupervised salient-object detection built exactly on
    DINO features (Wang et al., 2022). It gives a published reference point and needs no cluster merging at all.
- [ ] **A4.5 Cluster scoring.**
  * **Approach:** for each cluster compute five features:
    1. border fraction,
    2. distance of its centroid from the image centre,
    3. compactness (area / convex hull area),
    4. relative size,
    5. contrast to the background prototype from A4.2.

    Combine them first with hand-set weights. Then fit the 5 weights with logistic regression on the A3 folds.
  * **Reason to try it:** each failure image shows a different cue would have worked (centre for `usod_test205`, contrast
    for `usod_test174`). A learned combination of five numbers is still nearly label-free.
  * **Report it separately:** the fitted version uses a little supervision.
* **Done when:** all five methods are in the `src/analyze_results.py` comparison, with confidence intervals and paired tests against the border rule.
  We keep the best one as the new default selector.

### A5. Better features: registers, cosine distance, higher resolution (GPU, about 1 day)

* **What:** three changes to the features, each tested on its own.
* **Why:** these target the 34 cluster failures (mostly small objects) and the grid artifact seen in the worst images.
* **For:** raising the oracle (the ceiling), so better selection has more to work with. Target: **oracle at least 0.75, cluster failures at most 20**.
* **How and reasons:**
- [ ] **A5.1 `dinov2_vitb14_reg` (registers).**
  * **Reason:** the "Vision Transformers Need Registers" paper shows that DINOv2 creates high-norm artifact tokens in
    low-information areas such as open water, and that register tokens remove them. That matches the checkerboard
    texture in our worst cases.
  * **How:** change one argument (`--model dinov2_vitb14_reg`); the patch tokens start after the 4 register tokens (handled in A1).
- [ ] **A5.2 L2-normalise features before k-means (cosine k-means).**
  * **Reason:** DINO features are usually compared with cosine similarity. Plain Euclidean k-means also reacts to feature
    *norm*, and the norm is exactly what the artifact tokens distort.
  * **How:** `X = X / np.linalg.norm(X, axis=1, keepdims=True)` before clustering.
- [ ] **A5.3 Higher input resolution in one pass (644x868, about 46x62 patches).**
  * **Reason:** stride 3 did not help because each token still sees a 14 px patch of the same 448x588 image. A larger input
    makes each patch cover a smaller part of the scene, which is what small objects need. It is also far cheaper than 25 shifted passes.
  * **How:** `--height 644 --width 868` in A1. The PyTorch attention without xFormers handles about 2,900 tokens on 12 GB.
* **Done when:** each change has its own run in the comparison. We report the change in oracle IoU, border/selector IoU and the
  number of cluster failures in each object-size bin.

### A6. Linear probe: what does a tiny bit of supervision give? (GPU or CPU from the cache, about 1 day)

* **What:** train a single 1x1 convolution (768 inputs, 1 output) on the frozen patch features to predict the foreground.
* **Why:** it removes the selection problem completely (no clustering, no rule), and it costs about 1 ms per frame.
  Even if we prefer training-free methods, it tells us the realistic ceiling of these frozen features. Without it we cannot
  judge whether 0.60 from A4 is good or poor.
* **For:** a reference point for A4, and the most likely component of a deployable system (A7 and Part B).
* **How:**
  1. Inputs: cached `patch` features. Targets: ground-truth masks.
  2. Training: upsample the logits bilinearly to 480x640, then train with BCE + Dice loss, AdamW, lr 1e-3, about 50 epochs.
     It trains in seconds.
  3. Evaluation: 5-fold cross-validation using `splits.json`. Train on 240 images, test on 60, average over the folds.
  4. Second variant: a 2-layer MLP head (768 → 256 → 1).
* **Done when:** the cross-validated IoU with confidence intervals is in the comparison. Target: **at least 0.75**. If the probe is far above
  A4, plan to move the deployment path to a probe (B4, B5).

### A7. Metrics that match the AUV use case (CPU plus SUIM download, about 1 day)

* **What:** a new `eval_auv.py` that measures what the vehicle needs, not only pixel IoU.
* **Why:** every USOD image contains an object, so we have **never measured false alarms**. IoU also does not tell us
  whether the "object ahead" decision (README Sec. 4.2) would be right.
* **For:** knowing whether any of this is usable on an AUV. This is also the number a reviewer or supervisor will ask for first.
* **How:**
  1. **Negative set (empty water):**
     * Download SUIM. Its masks are colour-coded, with background water BW = black (0, 0, 0).
     * Keep the images whose mask is at least 98% BW.
     * Add water-only crops from USOD images, taken where the ground truth is empty.
  2. **Image-level detection:**
     * Score = fraction of foreground pixels in the central ROI (the middle third in both directions).
     * Report the AUROC between USOD images (positives) and the negative set.
     * Report the false-positive rate at the threshold that keeps 95% recall on USOD.
  3. **ROI IoU:** IoU restricted to the central ROI, since that is what an obstacle cue uses.
  4. Run it for the border rule, the best A4 selector and the A6 probe.
* **Done when:** a table with AUROC, the false-positive rate at 95% recall, and ROI IoU for the three methods.

---

## Part B: recommended extras (after Part A, or when a Part A result points here)

| ID | What | Why I recommend it (reason behind the approach) | How (short) | When to do it |
|---|---|---|---|---|
| B1 | **SAM / SAM 2 refinement** of the coarse mask | Our masks are blocky and miss thin parts (`usod_test219` starfish arms). SAM is very good at boundaries but needs a prompt; our mask gives one without labels | Prompt SAM with the mask's bounding box plus 1 to 3 positive points at the peaks of the distance transform; keep the SAM mask that best overlaps the coarse mask | After A4, if boundary errors dominate the remaining failures |
| B2 | **Dense CRF or bilateral solver** | Cheaper than SAM; snaps the mask to colour edges. TokenCut used a bilateral solver for the same reason | `pydensecrf` on the soft mask with the RGB image | If SAM is too slow for the vehicle |
| B3 | **Backbone sweep: ViT-S, L, g (with registers)** | Quantifies the speed vs accuracy trade-off. ViT-S is about 2.4x faster than B and may be enough for real time | `--model` in A1, the same evaluation | After A5, once the best recipe is known |
| B4 | **Train on SUIM, test on USOD** (binary, then 8 classes) | USOD is small (300 images, test-only). SUIM has 1,525 training images and semantic classes, which gives a proper train/test split and a check on cross-dataset generalisation | The A6 probe trained on SUIM "not water"; then an 8-class head | After A6 shows the probe works |
| B5 | **LoRA or last-block fine-tuning** | If the probe saturates, adapting the backbone to underwater colour cast and haze is the next step up | LoRA on the attention projections of the last 4 blocks, trained on SUIM | Only if A6 and B4 leave a clear gap |
| B6 | **SVAM-Net baseline** | A fair comparison with a method trained specifically for underwater salient-object detection on this very benchmark | Run the released checkpoints on USOD | Before writing up results |
| B7 | **Underwater enhancement before the backbone** (gray-world, CLAHE, FUnIE-GAN) | Tests whether the colour cast and haze hurt DINOv2. This is the domain-shift question that README Sec. 14 lists as untested | Pre-process the image, then the same pipeline | After A5 |
| B8 | **Multi-layer features** (last 4 blocks) | Earlier layers keep finer spatial detail, which may help small objects | `model.get_intermediate_layers(x, n=4)` in A1 | If A5.3 does not fix the small objects |
| B9 | **Real-time deployment**: ONNX to TensorRT fp16/INT8 on a Jetson, ViT-S at 224x294, probe head | The project target is 5 to 10 FPS on an embedded GPU; we have only desktop timings | Export, measure end to end on the Jetson | After the method is fixed (A4 to A6) |
| B10 | **Video and temporal smoothing** | Single-frame masks flicker; the "object ahead" rule needs stability over frames | Warm-start centroids, run the backbone every 2nd frame, smooth the masks | With B9 |
| B11 | **Sonar / DVL / depth fusion** | The camera gives bearing and extent but never range; safety needs a second sensor | Use the mask to gate the forward-sonar returns (design first) | Long term |
| B12 | **Verify the dataset licences and pick a licence for our scripts** | README Sec. 18 says these are unverified; this is needed before publishing masks or code | Check the IRVLab pages | Before any public release |

---

## Part C: things we stop doing (and why)

| Stop | Evidence | Instead |
|---|---|---|
| Stride-3 overlapping windows | No significant IoU change for 768-D (p = 0.06), 17x more backbone time, 2 s PCA per image | Higher input resolution (A5.3) |
| Clustering on 3-D PCA-RGB | Worse than 768-D in every setting (−0.03 to −0.10 IoU, p < 0.001) | Keep PCA-RGB only as a visualisation |
| Raising k to get better numbers | k up to 6 raises the oracle (+0.028) but lowers the deployable score (−0.029), both significant | Fix the selector (A4) before adding clusters |
| Quoting oracle numbers as performance | They use the ground truth | Always report the label-free number next to it |

---

## Decision points (what each result changes)

* **After A4:** if the best selector reaches about 0.60 or more, it becomes the default and A6 is mainly a reference. If no selector
  beats about 0.56 (the E2 768-D border rule), the training-free route is close to its limit; move the main effort to A6 and B4.
* **After A5:** if registers remove the grid artifact and cut the cluster failures, use `_reg` models everywhere from then on.
* **After A6:** if the probe is at least 0.10 IoU above the best training-free method, the deployable system should be "frozen
  backbone + probe", and Part B focuses on SUIM (B4) and Jetson speed (B9).
* **After A7:** if the false-positive rate on empty water is high for every method, add an image-level "object present" gate
  using the CLS token (cached in A1) before trusting any mask.

## Suggested schedule

| Week | Tasks |
|---|---|
| 1 | A0, A1, A2, A3 (infrastructure: everything after this is fast) |
| 2 | A4 (selectors) and A5 (features); update README Sec. 10 with the new runs |
| 3 | A6 (probe) and A7 (AUV metrics); decide the deployable recipe |
| 4+ | Part B items chosen by the decision points above |
