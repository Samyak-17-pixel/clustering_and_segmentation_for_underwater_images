# Underwater Subject Segmentation with Frozen DINOv2

**Find the animal, diver or object in an underwater picture and draw its mask, with a pre-trained vision model that we hardly change.**
First without any labels (clustering), then with a small trained part on top, and finally on new footage from our own test tank.

![Project overview](docs/figures/01_overview.png)

*The whole project on one page. Part 1: no labels. Part 2: with labels. Part 3: test on new footage.*

| | |
|---|---|
| Course | Machine Learning for Ocean Engineers, Department of Naval Architecture and Ocean Engineering, IIT Madras |
| Team | Sanjay (NA23B069), Bharath (NA23B063), Sarthak (NA23B070), Samyak (NA23B007) |
| Code | Python, PyTorch; everything is in `src/` |
| Results | [`results.md`](results.md) (all numbers and pictures) |
| Recipe to rerun everything | [`docs/REPRODUCE.md`](docs/REPRODUCE.md) |
| Methods in plain language | [`docs/METHODS.md`](docs/METHODS.md) |
| Open work | [`docs/ROADMAP.md`](docs/ROADMAP.md) |

## Contents

1. [What this project does and why](#1-what-this-project-does-and-why)
2. [Results at a glance](#2-results-at-a-glance)
3. [Background from scratch](#3-background-from-scratch)
4. [The datasets](#4-the-datasets)
5. [The method, step by step](#5-the-method-step-by-step)
6. [Repository layout](#6-repository-layout)
7. [Environment](#7-environment)
8. [Reproduce everything](#8-reproduce-everything)
9. [Limitations and honest caveats](#9-limitations-and-honest-caveats)
10. [Future work](#10-future-work)
11. [Glossary](#11-glossary)
12. [References](#12-references)
13. [Changelog](#13-changelog)

---

## 1. What this project does and why

**The task.** Given an underwater photo, output a **mask**: a black and white picture of the same size where white marks the *subject* (a fish, a diver, a wreck, a robot) and black marks everything else (water, sea floor, tank walls). This is called **segmentation**.

**Why it matters.** An underwater vehicle (AUV) needs to know that "something is in front of me". A mask gives the position and size of that something and how they change from frame to frame, which can drive a slow-down or avoid decision (ideally fused with sonar or depth; this use is a proposal and has not been tested). Underwater images are hard: the water tints and blurs everything, light fades with depth, and labelled underwater data is scarce.

**The idea.** Instead of training a large segmentation network from scratch on a few hundred labelled images, we use **DINOv2**, a large vision model that was pre-trained on 142 million images without any labels. It gives each small square of a picture a *fingerprint* of 768 numbers that is similar for similar content. We keep the model **frozen** (unchanged) and ask two questions:

1. **Without labels:** can we group the fingerprints (clustering) and pick the subject groups with a simple rule?
2. **With labels:** how much does a small trained part on top of the frozen model add?

**Analogy for the whole project.** DINOv2 is an experienced photo editor who can describe every tile of a photo in detail. Part 1 asks the editor to sort the tiles into piles and then guesses which piles matter. Part 2 teaches an assistant, from a few hundred marked examples, which piles matter. Part 3 sends the assistant to a new building (our tank) to see whether the lesson still holds.

![Timeline](docs/figures/22_roadmap_timeline.png)

*Six steps: Review 1, Review 2, Gaussian mixtures, supervised stages, the subject rule, the tank test.*

## 2. Results at a glance

Mean IoU (overlap of our mask and the true mask; 1 is perfect) on images never used to choose any setting. Full tables and pictures: [`results.md`](results.md).

![Supervised results](docs/figures/14_bars_supervised.png)

*Supervised stages A to D against the best methods without labels, on USOD and SUIM test images.*

| Step | Result |
|---|---|
| First baselines (PCA mask / k-means, k = 2) on USOD | 0.42 / 0.49 |
| k-means, up to 6 clusters, no labels | 0.52 (the best possible choice of clusters would give 0.70) |
| Dendrogram, no labels | **0.65** |
| Gaussian mixtures, no labels | USOD 0.54 / 0.59, SUIM not-water 0.51 / **0.63** |
| Supervised A, B, C, D (USOD / SUIM not-water) | 0.80, 0.82, **0.87**, 0.87 / 0.88, 0.90, **0.92**, 0.92 |
| Train on USOD, test on SUIM saliency | 0.79 (best no-label method: 0.55) |
| Train on SUIM, test on USOD | 0.36, because the two datasets define "subject" differently; **0.78** after we fixed one rule |
| Tank frames (new footage), best model | fish IoU 0.81, but 53% of empty frames get a false alarm (provisional) |

**In one paragraph.** DINOv2's features already contain the subject; without labels we can cluster them well but cannot reliably *choose* which clusters are the subject (0.52 to 0.65). A tiny trained head fixes that (0.80 to 0.92). The head learns the annotation convention of its dataset, so we defined one subject rule. On a new site (our tank) the models find the fish but mark too much on empty water; that is the open problem.

## 3. Background from scratch

### 3.1 Image, patch, feature

A computer sees a picture as a grid of numbers. DINOv2 cuts the picture into **patches** (squares of 14 x 14 pixels) and turns each patch into a list of **768 numbers**, its *feature vector* or *fingerprint*. Patches that look or mean alike (two patches of blue water, two patches of fish skin) get similar fingerprints. To look at 768 numbers we squeeze them to three with **PCA** (principal component analysis: keep the three directions with most variation) and show them as red, green, blue.

![PCA-RGB maps](docs/figures/02_pca_rgb_stride14_vs_3.png)

*Left to right: photo, PCA-RGB map with non-overlapping patches, PCA-RGB map with overlapping windows (stride 3). Diver, water and reef come out in different colours although the model never saw an underwater-specific label.*

### 3.2 DINOv2 in detail

| Item | Value |
|---|---|
| Model | DINOv2 ViT-B/14 (a **vision transformer**, "base" size, patch size 14) |
| Size | about 86 million weights, 12 transformer blocks, 768 numbers per patch |
| Training | self-supervised (self-distillation, no labels) on 142 million curated images (Oquab et al., 2023) |
| Weights | downloaded by `torch.hub` (`facebookresearch/dinov2`, `dinov2_vitb14`, about 330 MB) |
| Our input | 448 x 588 pixels, i.e. a grid of 32 x 42 patches; with **stride 3** (see below) 145 x 192 patches |
| Why it suits our task | its patch features carry meaning (object parts, materials), not just colour; no underwater training is needed |

**Stride 3.** One pass of the model gives a fingerprint every 14 pixels. If we run the *unchanged* model on 25 copies of the image shifted by 0, 3, 6, 9 and 12 pixels in both directions and interleave the results, we get a fingerprint every 3 pixels: smoother maps and finer boundaries (it turned out to barely change the final scores, and costs 25 times more time).

### 3.3 IoU, the score

**IoU** (intersection over union) = (area where our mask and the true mask are both white) divided by (area where at least one is white). 1.0 = perfect, 0 = no overlap. A mask that covers the whole picture scores about 0.22 on USOD, where the subject is small.

### 3.4 Clustering without labels

**Clustering** sorts the patches into groups by similarity of their fingerprints, without knowing what the groups are. **Analogy:** sorting a bag of marbles into bowls so that marbles in a bowl look alike, without being told what the colours are called. But a segmentation needs one more decision: *which groups are the subject?* We call that the **selection** step.

### 3.5 Supervised learning

**Supervised** learning shows the model many images with their true masks and adjusts a small set of numbers (weights) until its masks match. We split the data into **train** (the model learns from it), **validation** (we choose settings such as the threshold) and **test** (the final score, used once).

## 4. The datasets

| Dataset | Content | How we use it |
|---|---|---|
| **USOD** | 300 underwater images (640 x 480) with a binary mask of the one salient object (Islam, Wang, Sattar) | the main dataset for the no-label study; split 160 train / 40 validation / 100 test for supervised work |
| **SUIM** | 1,525 train/validation and 110 test images with 8-class masks: water, divers, plants, wrecks, robots, reefs, fish, sand/rocks (Islam et al., IROS 2020) | second dataset; two derived ground truths (see below); split 1,375 / 150 / 110 |
| **Dhwani** | 302 frames from six GoPro clips of our indoor underwater test tank (murky green water, perforated panels, steel frames, floor debris, a school of fish) | test only, never used for training or tuning |

![Data splits](docs/figures/08_data_splits.png)

*Splits are fixed (seed 0). The test images are used once, for the final number.*

**Two ground truths from SUIM.** "Saliency" = SUIM's own salient-object mask (about 20 percent of the image; 108 usable images). "Not-water" = every pixel that is not water (about 60 percent; 92 usable test images, because 18 images contain almost no water). They are different questions: *which thing stands out?* against *what is not open water?*

**Ground truth of the tank frames.** We built the dataset ourselves (`src/dataset/build_gopro_dataset.py`): in-water time windows were tagged by eye (fish, floor, structure, waterline, empty), candidate frames were taken every 0.5 s, filtered for quality, and 302 were picked by farthest-point sampling on DINOv2 embeddings so that they differ from each other; empty water is capped at 30 percent. There are no human masks. For **19 fish frames** we drafted masks with SAM (Segment Anything) from boxes we placed by hand, **not yet reviewed**, so all tank numbers are provisional. The other **283 frames** contain no animal or object, so the correct mask is empty.

![Dhwani frames](docs/figures/18_dhwani_frames.png)

*The median frame of each tag in the tank dataset: fish, structure, floor, empty water.*

The datasets are not redistributed; see [`data/README.md`](data/README.md) for where to get them and the folder layout.

## 5. The method, step by step

The project grew in stages. Each stage below says what we did, why, and what came out. All masks shown are on the **same four test images**, chosen by a rule (the USOD test images closest to the median object size of the small, medium and large size groups, plus the SUIM test image closest to the median), so methods can be compared by eye. Details and code: [`docs/METHODS.md`](docs/METHODS.md).

### 5.1 Review 1: first baselines

**PCA mask.** Threshold the first principal component of the fingerprints. **k-means with 2 groups.** Sort the patches into two piles. In both, the pile that touches the image border is called water and the other is the subject (the **border rule**: photographers put the subject in the middle).
Result on 300 USOD images: **0.42** and **0.49**.

![Review 1 baselines](docs/figures/m1_review1_pca_kmeans2.png)

*Original image (top) and mask of each method, with its IoU printed on the mask.*

### 5.2 Review 2: k-means with more groups, and the oracle

A scene has water, floor, reef and maybe several objects, so we allow 2 to 6 groups (number chosen by the **silhouette score**, how well separated the groups are) and keep the border rule. Result: **0.52** on USOD (48 percent on SUIM not-water).

To know whether the *groups* or the *choice* is the problem, we measured an **oracle**: for each image try every way of calling groups subject or background and keep the best one (it uses the true mask, so it is not a real method). It scores **0.70**.

![Border rule](docs/figures/03_border_rule.png)

*The border rule on a toy patch grid: it needs no labels but fails if the subject touches the border.*

![Oracle versus border rule](docs/figures/a2_oracle_vs_border_rule.png)

*Each dot is one image: oracle IoU against border-rule IoU. The gap below the diagonal is what is lost by choosing the subject without labels.*

**Finding:** the groups are good, **the choice of the subject groups is the weak step.** Everything after this is about that choice.

### 5.3 Dendrogram

Instead of k-means we build a **tree**: start with every patch alone and repeatedly merge the two closest groups until one group remains (average linkage). **Analogy:** a family tree. We cut the tree where the jump between merge heights is largest (the **gap rule**), because that is where very different groups would be forced together, and again apply the border rule.

![Dendrogram](docs/figures/04_dendrogram_gap_rule.png)

*Toy data: three natural groups; the red dashed cut sits in the largest jump.*

Result (no labels): **0.65** on USOD (k-means 0.52) and 0.55 on SUIM saliency (0.43). The oracle hardly changes (0.70 to 0.71), so the gain comes from a better *choice*. It is not a uniform win: about one image in five gets worse.

![Dendrogram masks](docs/figures/m3_dendrogram.png)

*Dendrogram masks on the four example images.*

### 5.4 Gaussian mixture models

**Idea.** k-means gives each patch one hard label and assumes round clusters of equal size. A **Gaussian mixture** models every cluster as a bell-shaped cloud with its own centre, spread and weight, and gives each patch a *probability* for each cluster (the **responsibility**). **Analogy:** overlapping spotlights on a stage. It is fitted with the **EM algorithm**: guess the clouds, compute each patch's responsibilities (E-step), re-fit the clouds with them as weights (M-step), repeat. **BIC** is a score that rewards fit and penalises many clouds.

![GMM idea](docs/figures/05_gmm_concept.png)

*Mixture density, responsibility, EM and BIC in one picture.*

**Two variants.** (1) **Per-image:** the k-means pipeline with a mixture instead (PCA to 32 numbers, 2 to 6 Gaussians, border rule). (2) **Dataset-level:** one mixture (8 components) fitted once on patches of the training images; each component is labelled foreground or background by how often its patches touch the image border; a new image needs one pass and no fitting. All settings are chosen on the source dataset's training images only.

![BIC versus silhouette](docs/figures/06_bic_vs_silhouette.png)

*BIC picked the maximum number of components almost every time (DINOv2 features are not Gaussian) and scored 0.44; silhouette scored 0.59.*

Result: per-image 0.54 (USOD) and 0.51 (SUIM not-water); dataset-level 0.59 and **0.63** (best label-free on SUIM not-water). But it learns SUIM's idea of foreground and drops to 0.40 on USOD when fitted with SUIM.

![Comparison of unsupervised methods](docs/figures/11_compare_unsupervised.png)

*All unsupervised methods on the same four images (rows: original, ground truth, k-means, dendrogram, GMM per-image, GMM dataset-level).*

**Conclusion of Part 1.** Without labels the best method is the dendrogram (0.64 to 0.65 on USOD). All methods stay far below the oracle, and none keeps its gain when the label definition changes.

### 5.5 Supervised stages A to D

**Why.** The weak step is the choice of the subject. A labelled mask tells a model directly which patches are the subject. We keep DINOv2 frozen (86 million weights, fewer than 2,000 labelled images: retraining it would overfit; **analogy:** keep the experienced translator and teach only your own vocabulary) and train a small part on top.

![Supervised pipeline](docs/figures/07_supervised_pipeline.png)

*Blue = frozen, orange = trained. Output: a probability per pixel; the threshold is chosen on validation images.*

| Stage | What is trained | Size |
|---|---|---|
| **A** linear probe | one 1 x 1 layer: a weighted sum of a patch's 768 numbers plus a bias, then a sigmoid | 769 weights |
| **B** conv head | a small convolutional head; each patch also looks at its neighbours (up to 15 x 15) | about 2 million |
| **C** LoRA | head of B plus small corrections inside DINOv2 (attention layers and patch embedding of all 12 blocks, rank 16), end to end | 906,432 (about 1% of the backbone) |
| **D** SAM | nothing; SAM's sharp outlines replace ours when a rule chosen on validation accepts them | 0 |

![LoRA concept](docs/figures/09_lora_concept.png)

*LoRA leaves the big weight matrix W untouched and adds a small trainable correction B(Ax) of rank 16 (**analogy:** sticky notes on a textbook).*

**Training.** Loss = **BCE + Dice** (BCE checks every patch; Dice checks the overlap of the whole mask, so small objects are not ignored). Features for A and B are computed once and cached (heads train in minutes). Stage C: 3,000 steps in about 14 minutes with flips, crops, brightness/contrast/saturation and blue-green colour-cast augmentation. Settings and thresholds come from validation images only.

**Result (test images).** USOD A 0.80, B 0.82, **C 0.87**, D 0.87; SUIM not-water 0.88, 0.90, **0.92**, 0.92. The biggest step is the first (+0.16 on USOD): the features already contain the answer. LoRA helps most for small objects (0.72 to 0.85 below 10 percent of the image). **SAM adds nothing**, because the ground-truth masks are blobby annotations.

![By object size](docs/figures/17_iou_by_object_size.png)

*Gain from supervision by object size on USOD.*

![Comparison of supervised stages](docs/figures/12_compare_supervised.png)

*All stages on the same four images (rows: original, ground truth, A, B, C, D, unified model).*

### 5.6 The label problem and the one subject rule

**Test.** Train on USOD only and test on SUIM, and the reverse. USOD to SUIM saliency scores **0.79**. SUIM to USOD scores only **0.36**: SUIM labels everything that is not water, USOD only the salient object, so the SUIM-trained model marks about 74 percent of USOD pixels when the truth is 22 percent. **Analogy:** two graders with different marking schemes. The features transfer; the label definitions do not.

**The rule** (written down and used for both datasets):
1. animals, divers, robots and wrecks are the subject;
2. if the scene has only reef and water, the reef is the subject;
3. otherwise there is no subject and the correct mask is empty.

It is derived from SUIM's class masks and agrees with SUIM's own saliency masks at IoU 0.99. USOD already follows it.

![Subject rule](docs/figures/10_subject_rule.png)

*The three branches of the rule.*

![Cross-dataset results](docs/figures/15_cross_dataset.png)

*Cross-dataset results before and after the rule: SUIM to USOD rises from 0.36 to 0.78.*

With one label, a **unified model** (one output, one threshold) scores 0.84 on USOD and 0.84 on SUIM, 0.77 to 0.78 across datasets. Five seeds vary by about 0.01. On crops of empty water the **false-alarm rate** (a mask where nothing is) is 10 percent for model C against 100 percent for k-means, which must always output something.

![Unified model](docs/figures/m10_unified_model.png)

*Unified model C. In the SUIM image only the fish are the subject; the reef is background.*

### 5.7 Tank frames (new footage)

All models, trained only on USOD and SUIM, are run on the 302 Dhwani frames. We also trained a **strict-rule** model (only animals and objects are subjects; reef, sand and tank walls are background) with **underwater augmentation** (haze, darkening, colour cast, contrast loss, blur; strengths fixed before training) and **no-subject crops** (15 percent of training crops contain no subject).

![Tank results](docs/figures/16_tank_results.png)

*Fish IoU on the 19 fish frames (left, higher is better) and false-alarm rate on the 283 empty frames (right, lower is better). Provisional until the draft masks are reviewed.*

* **Unsupervised methods cannot say "nothing here":** k-means marks something in 100 percent of empty frames, the dendrogram in 94 percent; fish IoU 0.18 and 0.47.
* **Supervised models find the fish** (recall 0.98 to 0.99). The earlier salient-channel model is best (fish IoU 0.81) but still raises a false alarm on 53 percent of empty frames. The unified model gives 0.48 and 76 percent. The strict-rule model halves the false alarms of Stage C (76 to 41 percent) but marks the murky water around the fish (fish IoU 0.25).
* Reading: on a new site the simplest, least adapted model is the most selective; pooling two label conventions and adapting the backbone made things worse.

![Tank, supervised](docs/figures/20_dhwani_supervised.png)

*Supervised models on four tank frames.*

We also rendered videos of all 12 methods on six 30-second tank segments (`src/dataset/make_videos.py`); the predictions are made frame by frame, so masks flicker slightly.

## 6. Repository layout

```
.
├── README.md             this file
├── results.md            all results with tables and pictures
├── requirements.txt      pinned package versions
├── CLAUDE.md  LICENSE
├── src/
│   ├── common/           gt_utils.py: ground truth loading (USOD, SUIM saliency / not-water / subject / strict)
│   ├── kmeans/           dinov2_seg.py (Review 1), dinov2_cluster_union.py, dinov2_cluster_union_stride.py (Review 2, SUIM)
│   ├── dendrogram/       dendro_cluster.py, dinov2_dendro.py, summarize_dendro.py
│   ├── gmm/              gmm_common.py, gmm_dev.py, gmm_fit_global.py, gmm_eval.py, compare_gmm.py
│   ├── supervised/       features, labels, train_head (A, B), finetune (C), sam_refine (D), compare*, cross_compare,
│   │                     false_alarm, seeds, tta, hard_cases, uw_aug, export_examples, worked_example
│   ├── dataset/          build_gopro_dataset, sam_draft, run_on_gopro, tank_eval, make_videos
│   ├── analysis/         analyze_results.py: statistics and figures on the k-means result tables
│   └── figures/          make_readme_figures.py: every picture in docs/figures
├── docs/
│   ├── METHODS.md        every method in plain language
│   ├── REPRODUCE.md      commands in order, run times, expected numbers
│   ├── ROADMAP.md        what is done and what is open
│   └── figures/          the pictures used in the .md files
├── data/README.md        where to get the datasets and how to lay them out   (data itself is not in git)
└── presentations/        build scripts of the review slide decks (decks themselves are not in git)
```

Not in the repository (created locally by the scripts or supplied by you): `data/` (datasets), `videos/` (raw footage), `outputs/` (feature caches, checkpoints, predictions) and `results/` (tables, collages, videos).

![Data flow](docs/figures/21_data_flow.png)

*Which script produces what.*

## 7. Environment

| Item | Version used |
|---|---|
| OS, GPU | Linux 6.8, NVIDIA RTX A2000 with 12 GB |
| Python | 3.10 |
| PyTorch / torchvision | 2.14.1 / 0.29.1 (CUDA build) |
| numpy, scipy, scikit-learn, pandas | 2.2.6, 1.15.3, 1.7.2, 2.3.3 |
| pillow, matplotlib, opencv-python-headless, tqdm, tabulate | 12.3.0, 3.10.9, 5.0.0.93, 4.70.1, 0.10.0 |
| peft (LoRA), transformers (SAM) | 0.21.2, 5.20.2 |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Weights (DINOv2 and `facebook/sam-vit-base`) download automatically on first use. About 15 GB of disk is needed for the intermediate files.

## 8. Reproduce everything

![Run order](docs/figures/24_repro_order.png)

*Nine steps; the full commands, run times and the numbers to expect are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).*

Short version, from the repository root, after placing the datasets as described in `data/README.md`:

```bash
python src/kmeans/dinov2_cluster_union_stride.py --stride 3 --kmax 6        # k-means (about 30 min)
python src/dendrogram/dinov2_dendro.py --data USOD --gts usod --stride 3     # dendrogram (about 45 min)
python src/gmm/gmm_dev.py usod && python src/gmm/gmm_fit_global.py usod     # Gaussian mixtures (then suim, gmm_eval, compare_gmm)
cd src/supervised && python features.py && python train_head.py A && python train_head.py B && python finetune.py 3000
```

Random seeds are fixed (k-means and silhouette `random_state=0`, a seeded split); GPU half precision can change the last digit between machines.

## 9. Limitations and honest caveats

* **The tank ground truth is provisional:** 19 fish masks are SAM drafts that have not been reviewed, and 283 frames are *assumed* to contain no animal or object. The large differences between methods (fish IoU 0.81 against 0.25) do not depend on small mask corrections, but the exact values may move.
* **Label definitions decide the score.** "Subject" is a human decision. The SUIM subject label is derived from SUIM's own class masks, so SUIM scores follow SUIM's convention. The tank study used a stricter rule than the one used for the earlier results.
* **The oracle is not a method.** It uses the true mask and only measures what the clusters could do.
* **Selection bias in one place:** the best dendrogram variant was chosen on the same images it is reported on, so its gain is a little optimistic.
* **Small validation sets:** USOD validation has 40 images; most results are a single run (seeds are reported where run).
* **Not tested:** real-time speed on an embedded vehicle GPU (desktop timings only), temporal smoothing, turbid water beyond the tank, other camera types.
* **SVAM-Net**, the baseline published with USOD, was not run (executing its downloaded upstream code was not permitted in this project).
* **History:** older commits on GitHub still contain files from before this clean-up (USOD images and k-means result files); the current version does not.

## 10. Future work

1. Review the 19 draft fish masks and confirm the 283 empty frames (see `docs/ROADMAP.md`).
2. Reduce false alarms on empty water: an image-level "is there a subject?" decision, more no-subject training crops, smoothing over time in video.
3. More varied labelled underwater data, or a few labelled tank frames for fine-tuning.
4. Separate the individual subjects (instance masks) instead of one mask for all.
5. Measure and optimise run time on the vehicle's embedded GPU (smaller backbone, lower resolution, a linear probe without clustering).

## 11. Glossary

| Term | Meaning |
|---|---|
| **Patch** | a 14 x 14 pixel square of the picture |
| **Feature / fingerprint** | the 768 numbers DINOv2 gives to one patch |
| **DINOv2 ViT-B/14** | the pre-trained vision transformer we use (86 M weights) |
| **Frozen** | not changed by our training |
| **PCA** | squeezes many numbers per patch to a few directions that keep most of the variation |
| **Mask** | black and white picture; white = subject |
| **IoU** | overlap of two masks divided by their combined area |
| **Ground truth** | the reference mask made by a person or by a rule |
| **Cluster / k-means** | a group of similar patches / an algorithm that sorts patches into k groups |
| **Silhouette** | score for how well separated clusters are; used to choose k |
| **Border rule** | groups touching the image border are background |
| **Oracle** | choosing groups with the true mask: a ceiling, not a method |
| **Dendrogram, linkage** | a tree of merges; linkage = how distance between groups is measured |
| **Gap rule** | cut the tree at the largest jump between merge heights |
| **GMM** | Gaussian mixture model: clusters as bell-shaped clouds |
| **Responsibility** | probability that a patch belongs to a cluster |
| **EM** | the guess-and-adjust loop that fits a mixture |
| **BIC** | fit-versus-complexity score used to choose the number of clusters |
| **Supervised** | learning from labelled masks |
| **Linear probe** | a single trained layer on frozen features |
| **Convolutional head** | small network where each patch looks at its neighbours |
| **LoRA** | small low-rank corrections added next to frozen weights |
| **BCE + Dice** | per-patch loss plus an overlap loss that protects small objects |
| **SAM** | Segment Anything Model: draws sharp outlines from a hint box or point |
| **Subject rule** | our definition of what counts as the subject |
| **False alarm** | a mask drawn where there is no subject |
| **TTA** | test-time augmentation: average predictions on flipped / rescaled copies |
| **Domain shift** | test images look different from the training images |
| **Train / validation / test** | learn from / choose settings with / final score |

## 12. References

1. M. Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023 (arXiv:2304.07193).
2. A. Dosovitskiy et al., *An Image is Worth 16x16 Words* (Vision Transformer), ICLR 2021.
3. M. J. Islam, R. Wang, J. Sattar, *SVAM: Saliency-guided Visual Attention Modeling by Autonomous Underwater Robots* (USOD dataset), arXiv:2011.06252.
4. M. J. Islam et al., *Semantic Segmentation of Underwater Imagery: Dataset and Benchmark* (SUIM), IROS 2020, arXiv:2004.01241.
5. E. J. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
6. A. Kirillov et al., *Segment Anything*, ICCV 2023.
7. A. P. Dempster, N. M. Laird, D. B. Rubin, *Maximum likelihood from incomplete data via the EM algorithm*, 1977.
8. G. Schwarz, *Estimating the dimension of a model* (BIC), 1978.
9. P. J. Rousseeuw, *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*, 1987.

DINOv2 and SAM weights are used under their own licences; the datasets under the terms of their authors (the USOD licence was not verified; check the project page before redistributing).

## 13. Changelog

| Date | Change |
|---|---|
| 2026-11-05 | v3.0: repository reduced to code and methodology; new README, `results.md`, `docs/METHODS.md`, `docs/REPRODUCE.md`, `docs/ROADMAP.md`, `requirements.txt` and 40 explanatory figures; datasets, result folders, decks and videos are no longer versioned |
| 2026-11-05 | v2.0: project status brought up to date (Gaussian mixtures, supervised stages, cross-dataset test, subject rule, tank dataset, Review 3 deck) |
| earlier | v1.x: k-means study, dendrogram study, SUIM, supervised stages (see `git log`) |
