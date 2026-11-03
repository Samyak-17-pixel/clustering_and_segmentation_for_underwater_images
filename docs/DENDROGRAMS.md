# Dendrograms in simple words

Everything we built with dendrograms (hierarchical clustering) for training-free underwater object segmentation, explained without jargon. Code is in `src/dendrogram/`, results in `results/dendrogram/`.

Full technical write-up with tables, statistics and caveats: [README Sec. 19](../README.md#19-dendrogram-hierarchical-clustering-instead-of-k-means).

---

## 1. The problem we were solving

Our pipeline looks at an underwater photo with a frozen DINOv2 model. DINOv2 cuts the photo into small squares called **patches** and gives each patch a list of 768 numbers (its "features"). Patches that look alike (e.g. all water, or all fish) get similar numbers.

We then **group** the patches and decide which groups are the object and which are water. Our earlier method was **k-means**:

- You must say in advance how many groups you want (k).
- If k is wrong, the object gets split into pieces or merged with water.
- Choosing the best k and the best groups is hard without the answer key.

**Idea:** instead of fixing k, build a *tree* of groupings once, then look at it at different heights. That tree is a dendrogram.

---

## 2. What is a dendrogram?

Think of a family tree built from the bottom up:

1. Start with every patch alone.
2. Repeatedly **merge the two most similar groups** into one.
3. Keep going until everything is one big group.

The record of all merges is the **dendrogram**. Each merge has a **height**: how different the two groups were when they were joined (low = very similar, high = quite different).

**Cutting the tree:** draw a horizontal line across the tree. The branches it crosses are your groups. Cut high → few big groups (k=2). Cut lower → more, smaller groups (k=3, 4, ...). So one tree gives *every* k at once, and the groups at k+1 are always a split of one group at k.

---

## 3. The four ways to merge ("linkage")

### 3.1 Why we need a "linkage" at all

Step 2 of the tree-building says "merge the two most similar groups". Measuring how different two *single patches* are is easy (the distance between their 768 numbers). But once groups hold many patches, "how far apart are group A and group B?" has no single obvious answer. Should we look at the closest members? The furthest? The average? The group centres? Each choice is a **linkage**, and each makes a different tree from the same patches, so a different set of groups.

### 3.2 A tiny example you can check by hand

Imagine patches as dots on a line (real patches live in 768 dimensions, but the idea is the same). Two groups:

- Group A = dots at **0** and **1**
- Group B = dots at **4** and **6**

| Linkage | What it measures | Value here |
|---|---|---|
| **Single** | closest pair: 1 and 4 | **3** |
| **Complete** | furthest pair: 0 and 6 | **6** |
| **Average** | mean of all 4 pairs: (4 + 6 + 3 + 5) / 4 | **4.5** |
| **Ward** | how much the total "spread inside groups" grows if A and B are merged (the spread is 0.5 + 2 = 2.5 before and 22.75 after) | **20.25** |

Same two groups, four different answers. The tree always merges the pair of groups with the *smallest* value, so the linkage decides who gets merged first.

### 3.3 The four linkages one by one

**Single linkage – "nearest neighbour wins"**
- Distance between groups = distance between their two *closest* patches.
- Two groups join as soon as *one* patch of each is close, even if the rest are far apart.
- This causes **chaining**: a row of dots 0, 1, 2, 3, 4, 5 each one step apart all join into one group, even though 0 and 5 are far apart.
- Good for: long thin shapes. Bad for us: water patches shade smoothly into fish patches, so a chain of in-between patches links everything together.
- What happens in practice: almost every patch joins one giant group, and the top split only peels off a few odd outlier patches.

**Complete linkage – "the worst pair counts"**
- Distance between groups = distance between their two *furthest* patches.
- Two groups join only if *every* patch of one is reasonably close to every patch of the other.
- Produces tight, round, similar-diameter groups. Very cautious about merging, so a single outlier patch can keep two otherwise similar groups apart.
- In practice: reasonable but a bit worse than Average because its cuts are driven by outliers.

**Average linkage – "typical pair"**
- Distance between groups = average over all pairs (one patch from each).
- A compromise between the two extremes above: one close pair does not glue groups together (unlike Single) and one far outlier does not block a merge (unlike Complete).
- This is the one that works best for us: the object patches form a clear group, the water patches form another, and the merge heights show a clear jump between "natural" groups, which is exactly what the **gap rule** (Section 5) looks for.
- On 768-D features we use **cosine distance** (do two patches point in the same direction?) because DINOv2 features are about direction more than size.

**Ward linkage – "keep groups compact"**
- Merge the two groups whose union increases the total within-group spread the *least* (the same quantity k-means tries to minimise).
- Like k-means, it likes groups of similar size and round shape, and it can split a large uniform area (open water) into several pieces rather than keep it whole.
- Needs ordinary (Euclidean) distance, so it is the only one that uses Euclidean on the 768-D features.
- In practice: clearly better than k-means's choices but behind Average, because big water regions get split.

### 3.4 What we measured (USOD, stride 3, 768-D, free tree, mean IoU over 300 images)

| Linkage | Silhouette cut | Gap cut | Oracle (best possible union) |
|---|---|---|---|
| Ward | 0.571 | 0.544 | 0.716 |
| **Average** | **0.630** | **0.646** | 0.711 |
| Complete | 0.568 | 0.527 | 0.676 |
| Single | 0.211 | 0.224 | 0.224 |

Reading the table:
- **Average is best** with both cut rules, and the gap cut suits it particularly well.
- **Ward and Complete** are similar to each other and a little above k-means (0.523).
- **Single** is far below even the trivial "everything is object" score (0.224); even the oracle cannot rescue it (0.224), because the tree never separates the fish into its own group in the first place.
- The oracle column shows that Ward's tree actually contains slightly better groups (0.716) than Average's (0.711). Average wins because it is easier to *choose* the right cut on its tree, not because its groups are better.

### 3.5 Why Single fails, on one real image

On our example fish image (usod_test153), single linkage put 27,759 of 27,840 patches (99.7%) in one group. The tiny leftover group sat mostly on the image border, so it was called water, and the huge group (slightly less border than average) was called object. The whole picture came out white, so the score was just the fish's share of the picture (0.177, shown as 0.18). Average linkage on the same image marked only 22.4% of patches (the fish is 17.7%).

### 3.6 Which one to use

Use **Average linkage** (cosine distance, 768-D features). Ward is a safe second choice. Avoid Single linkage for this task.

---

## 4. Variants we compared (16 in total)

- **Features:** the full 768-number DINOv2 features, or just 3 numbers per patch (PCA-RGB, a colour-like summary).
- **Tree type:**
  - *Free*: any similar patches may merge, wherever they are.
  - *Constrained*: only **neighbouring** patches in the image may merge (so groups are connected blobs).
- **Linkage:** Ward / Average / Complete / Single.
- 2 × 2 × 4 = **16 variants**, each tested at **stride 14** (non-overlapping patches, 1,344 per image) and **stride 3** (overlapping patches, ~27,840 per image, finer masks).
- Distance: Ward needs ordinary (Euclidean) distance; the others use cosine distance on 768-D features (Euclidean on PCA-RGB).
- **Speed trick:** a tree on 27,840 patches is too big, so for free trees at stride 3 we build the tree on **3,000 randomly sampled patches** and give every other patch the group of its nearest sampled patch.

Code: `src/dendrogram/dendro_cluster.py` (`build_tree`, `cut_labels`), `src/dendrogram/dinov2_dendro.py` (the full run).

---

## 5. Turning the tree into an object mask (without the answer key)

After cutting, we still must say which groups are the object. We tried four **label-free rules** (never look at the ground-truth mask):

1. **sil_border** – pick k (2–6) with the best *silhouette* score (how well separated the groups are), then call the groups with *few patches on the image border* the object. Rationale: objects are usually in the middle, water touches the edges.
2. **gap_border** – pick k using the **biggest gap in merge heights**. If the tree jumps a lot between k and k+1 groups, that is the natural number of groups. Then apply the same border test.
3. **subtree** – walk the tree from the top: branches with almost no border patches are object, border-heavy ones are water, mixed ones are split further.
4. **score** – give every branch a score (low border share, near centre, compact shape, plausible size) and keep the best-scoring ones.

Fixed constants in these rules were set up front, not tuned on the ground truth.

We also compute an **oracle**: the best possible union of groups, found by looking at the ground truth. It is *not* a usable method; it tells us the ceiling of what the grouping could achieve.

**Metric:** IoU (overlap divided by total area of predicted ∪ true object; 1 = perfect). Also F-measure and MAE are stored.

---

## 6. Datasets

- **USOD** (300 test images) – main benchmark.
- **SUIM test, saliency GT** (108 images) – object vs everything else.
- **SUIM test, not-water GT** (92 images) – almost everything that is not water counts as object, so "all white" already scores 0.52; this set can't show improvements well.

Trivial "everything is object" IoU for reference: USOD 0.224, SUIM-Saliency 0.199, SUIM-not-water 0.520.

---

## 7. Results (USOD, stride 3, 300 images)

| Method | Mean IoU |
|---|---|
| k-means baseline (k ≤ 6, border rule) | 0.523 |
| Ward + silhouette cut | 0.571 |
| Complete + silhouette cut | 0.568 |
| Average + silhouette cut | 0.630 |
| **Average + gap cut (best)** | **0.646** |
| Single linkage | 0.21 (fails) |
| Oracle (best possible union) | 0.711 |

- **Best variant:** 768-D features, free tree, Average linkage, gap rule.
- Gain over k-means: **+0.123 IoU** (95% CI +0.095 to +0.151, p ≈ 5e-14).
- Better than k-means on **52%** of images, worse on **20%**.
- SUIM-Saliency: similar gain (**+0.122**).
- SUIM-not-water: no meaningful gain (the metric is nearly saturated).
- `subtree` and `score` rules were worse than the border rule (~0.38–0.50), so we dropped them.
- Constrained trees (neighbours only) were not better than free trees.

Why it helps: the tree does not force k in advance, and Average linkage with the gap rule often finds the "natural" split between object and water; k-means with the silhouette rule often picks a wrong k.

---

## 8. Honest limitations

- **Best of 64 chosen on the same images** (16 variants × 4 rules), so 0.646 is slightly optimistic. A held-out split would be fairer.
- **Gap rule fails on ~20% of images.** Typical failure (usod_test40): a tiny group of outlier patches splits off first, so the biggest gap says "k = 2" and the mask is wrong (IoU 0.21), while silhouette would have chosen k = 4 (IoU 0.83). A **minimum-group-size** condition is the obvious fix, but untried.
- **Selection is still the bottleneck:** the oracle (0.711) shows good groups exist; the rules just don't always pick them.
- "More groups is better" (k = 6 over k = 4) is only true for the oracle; with a realistic rule k-means got slightly worse (0.560 → 0.531).
- SUIM-not-water is close to trivial.
- Dendrograms are drawn with only the top ~12 branches (`truncate_mode`), but all patches are in the tree.

---

## 9. Files

| Path | Purpose |
|---|---|
| `src/dendrogram/dendro_cluster.py` | tree building, cutting, the four rules, silhouette, GPU helpers |
| `src/dendrogram/dinov2_dendro.py` | runs all variants over a dataset, caches masks, writes metrics and figures |
| `src/dendrogram/summarize_dendro.py` | tables, paired statistics vs k-means, best-variant panels |
| `results/dendrogram/{usod,suim_saliency,suim_nonwater}/` | `metrics_s*.csv`, `summary_s*.md`, `stats_s*.csv`, `dendrograms/`, `compare/`, `panels_best/` |
| `results/dendrogram/_cache/` | cached cut masks per image |

## 10. How to run

```bash
source /home/vallabh/Research/Samyak/.venv/bin/activate

# build trees, cut, score (both strides, USOD)
python src/dendrogram/dinov2_dendro.py --data USOD --gts usod --stride 14
python src/dendrogram/dinov2_dendro.py --data USOD --gts usod --stride 3

# tables, statistics vs k-means, best-variant panels
python src/dendrogram/summarize_dendro.py --gt usod --strides 14 3
```

SUIM: add `--data data/SUIM/TEST --gts suim_saliency suim_nonwater` and run `src/dendrogram/summarize_dendro.py` with `--gt suim_saliency` / `--gt suim_nonwater`. Total runtime for everything was about 73 minutes.

## 11. One-paragraph takeaway

A dendrogram builds the grouping once and lets us pick the number of groups afterwards. Using DINOv2 features, Average linkage, and the "biggest gap" rule to pick where to cut, we raised the mean IoU on USOD from 0.52 (k-means) to 0.65, with a ceiling of 0.71 if the best groups could always be chosen. The remaining work is a more reliable cut rule, such as requiring a minimum group size, before the final review.
