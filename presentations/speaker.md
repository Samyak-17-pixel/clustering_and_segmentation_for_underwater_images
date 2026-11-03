# Speaker notes: Project Review 1

> **Which part is current:** the notes up to the line "Review 2 deck: speaker notes" belong to the old Review 1 deck (9 slides). For Review 2 (16 slides) jump to "# Review 2 deck: speaker notes (slides 3-16)".

Deck: *Unsupervised Underwater Object Segmentation Using Pretrained Vision Transformer Features* (9 slides, `presentations/…pptx.pdf`).
Suggested pace: about 8–10 minutes, roughly 1 minute per content slide, then questions.

Each slide has: **What it says**, **Explanation** (so you understand it), **Speaker notes** (say this), and **Likely questions**.

---

## Fix these before presenting

1. **Slide 7 has leftover instruction text.** It reads "Comparisons against appropriate baselines and sensitivity analysis are needed. Do not invent numerical results or claim superiority before experiments are performed." That is a note-to-self from a prompt, not slide content. Delete it. Keep the sentence before it.
2. **Slide 5 contradicts itself on the dataset.** The top says SUIM is the primary dataset and USOD expands it. The bottom box says the initial evaluation uses USOD, with SUIM for later. Everything done so far is on USOD (300 images, binary masks). Rewrite the top as: "USOD is the dataset used in the initial evaluation. SUIM is planned for later generalisation testing."
3. **Slide 5 samples.** Say which dataset the seadragon image and mask come from. If you are not sure, say "an example sample" and don't name the dataset.
4. **Slide 8 image.** The mask on the right has no caption. Add one, such as "example predicted mask", or remove it. A reviewer will ask what it is.
5. **Slide 1 title vs your project name.** The deck says "Unsupervised", the README says "Training-free". Both are defensible, but pick one wording and explain it once (see slide 1).

---

## Slide 1: Title ("Project Review 1")

**What it says:** Title only. A fish over a reef is the background image.

**Explanation:** The title has three ideas.
- *Unsupervised / training-free:* no labels are used and no network is trained.
- *Pretrained vision transformer features:* DINOv2 learned general image features from huge amounts of unlabelled images.
- *Object segmentation:* decide, per pixel, whether it belongs to an object or to the background.

**Speaker notes:**
"Good morning. My project is about finding objects in underwater images without training a new model. I reuse features from a pretrained vision transformer called DINOv2, group them with simple clustering, and turn that into a foreground and background mask. This is Review 1, so I will cover the problem, why it matters, the dataset and my methodology. Results come in the later reviews."

**Likely questions:** "What do you mean by unsupervised?" Answer: no labels are used to produce the masks. Labels are used only afterwards to measure how good the masks are.

---

## Slide 2: Background and problem statement

**What it says:** Objective, challenges, proposed approach, research question and scope. Two reef and fish photos sit on the right.

**Explanation:**
- *Objective:* given one underwater image, separate the foreground object(s) from the background.
- *Challenges:* water absorbs and scatters light, so images have colour cast, haze, low contrast and uneven lighting. Labelled underwater data is also scarce and expensive.
- *Approach:* use DINOv2 features as they are, with no fine-tuning.
- *Research question:* can DINOv2 features plus clustering plus a rule that needs no labels give useful binary masks?
- *Scope:* **binary** foreground/background, not semantic segmentation. The model does not say "this is a fish", only "this is an object".

**Speaker notes:**
"The task is simple to state. Given an underwater image, mark which pixels belong to the object and which to the water or background. It is hard underwater because light is absorbed and scattered, so images are hazy, colours shift to blue or green, and lighting is uneven. On top of that, labelled underwater data is scarce, so training a segmentation network from scratch is costly. My idea is to reuse DINOv2, which was pretrained on a huge collection of general images, keep it frozen, and see how far unsupervised clustering can take us. The research question is on the slide. I want to stress the scope: this is binary salient-object segmentation, not semantic segmentation. I am not labelling classes."

**Likely questions:**
- "Why not just train a U-Net?" Answer: it needs many pixel labels, which underwater data lacks, and I want to see how far free pretrained features go.
- "What does salient mean?" Answer: the object that stands out to a viewer, usually the main object in the scene.

---

## Slide 3: Relevance to computer vision and underwater robotics

**What it says:** Three blocks (computer vision, underwater perception, AUVs) with a submarine photo and a diver photo. A box at the bottom limits the claims.

**Explanation:**
- *Computer vision:* tests whether features learned on everyday images transfer to underwater images with no extra training. It also covers unsupervised segmentation by clustering patches, foreground selection, and converting patch-level results into pixel masks.
- *Underwater perception:* poor visibility, complex backgrounds and little data.
- *AUVs:* an autonomous underwater vehicle can use a foreground mask to notice structures or obstacles in its camera view.
- The **bottom box is deliberate honesty**: this project evaluates segmentation. It does not yet give metric depth, reliable obstacle avoidance or autonomous docking. The AUV link is motivation and a possible application.

**Speaker notes:**
"There are two reasons this matters. For computer vision, it tests whether features learned on everyday images transfer to a very different domain, underwater, without any retraining. For underwater robotics, a vehicle that can tell where an object is in its camera view has a first step towards awareness of what is ahead of it, for inspection, following a diver or spotting obstacles. I want to be clear about what I am claiming. This project measures segmentation quality. It does not give distance, it is not a certified obstacle-avoidance system, and it does not do docking. The AUV is the motivation and the future application."

**Likely questions:**
- "How would the AUV use the mask?" Answer: a possible rule is the share of foreground pixels in the centre of the image, fused with sonar or DVL for range. This is proposed and untested.
- "How does it pick the right mask without labels?" Answer: with a label-free rule (slide 6). The current rule is simple and fails in some cases, and improving it is part of the work. Labels are used only to evaluate.

---

## Slide 4: Research objectives and scope

**What it says:** One primary objective and four blocks: feature extraction and a frozen backbone, feature clustering and cluster analysis, foreground identification and background separation, and evaluation (metrics, failure analysis, efficiency). A box at the bottom restates the scope.

**Explanation:**
- *Primary objective:* test whether this is feasible.
- *Frozen backbone:* the DINOv2 weights are never changed.
- *Cluster analysis:* study how the number of clusters and the clustering setup affect the result.
- *Label-free foreground identification:* choose which clusters are the object without ground truth.
- *Bottom box:* ground-truth masks are for evaluation and error analysis only. The **oracle** (picking the best cluster combination using ground truth) is reported separately as an *upper bound*, not as a result the method can reach on its own.

**Speaker notes:**
"My primary objective is a feasibility study: can a frozen DINOv2 and simple clustering produce useful object masks? I break it into four parts. First, extract patch features and keep the backbone frozen. Second, cluster the patches and study how clustering settings change the outcome. Third, decide which clusters are the foreground without ground truth. Fourth, measure with IoU, F-measure and MAE, study failure cases, and look at computation time. One point matters for honesty. Ground truth is used only to evaluate. If I ever pick clusters using ground truth, that is an oracle, an upper bound on what better selection could achieve, and I will report it separately from the real method."

**Likely questions:** "What is an oracle?" Answer: a selection that peeks at the ground truth to choose the best option. It shows the best the clusters could do, not what a deployed system gets.

---

## Slide 5: Dataset selection and understanding

**What it says:** SUIM as the primary dataset and USOD as an extension (see fix 2). Each sample is an RGB image plus a ground-truth mask. Three pictures: the RGB image (a weedy seadragon), the ground-truth mask, and a DINOv2 PCA visualisation. The bottom box says the initial evaluation uses USOD and SUIM comes later for generalisation.

**Explanation:**
- **USOD** (from the SVAM-Net paper, Islam, Wang and Sattar): 300 underwater images with binary masks of the salient object. This is what the project currently uses.
- **SUIM** (Islam et al., IROS 2020): 1,525 training/validation images plus 110 test images, with 8 semantic classes. It will be used in later iterations. Its labels are semantic, so it needs converting to binary foreground/background.
- *Right picture:* the first three principal components of the 768-D patch features, shown as red, green and blue. Similar-looking regions get similar colours. The seadragon shows up as a different colour from the water and weeds, which is why clustering has a chance.
- The blocky look is because each DINOv2 patch covers 14×14 pixels.

**Speaker notes:**
"Each sample is an underwater RGB image and a ground-truth mask that marks the foreground. The image is the only input to the method. The mask is only the reference for scoring. I start with USOD, which has 300 underwater images with binary object masks, and I plan to add SUIM later to test generalisation. On the right is what DINOv2 sees: I compressed its 768-dimensional features into three colours. Notice the seadragon already stands out from the weeds and water. That is why clustering these features can separate the object. The blocks appear because DINOv2 looks at the image in 14 by 14 pixel patches."

**Likely questions:**
- "Why USOD first?" Answer: it already has binary masks, which match the task, so no conversion is needed.
- "Is 300 images enough?" Answer: yes for evaluation, since nothing is trained. It does limit how general the conclusions are, so SUIM comes next.

---

## Slide 6: Proposed methodology

**What it says:** Six steps, numbered 01 to 06, and a summary box.

**Explanation (step by step):**
1. **Preprocessing:** resize the image (448×588) and normalise with the ImageNet mean and standard deviation that DINOv2 expects.
2. **Feature extraction:** DINOv2 ViT-B/14, frozen, split the image into 14×14 patches. A 588×448 image gives 42×32 = **1,344 patches**, each described by a **768-number** vector.
3. **Feature analysis:** PCA to 3 dimensions *for visualisation only*. The original 768-D features are kept for clustering, because compressing them throws away information.
4. **Clustering:** K-means groups similar patches. The number of clusters is studied (the silhouette score can choose it).
5. **Foreground selection:** the baseline rule is a border heuristic. A cluster is called foreground if it touches the image border less than average, since objects tend to sit away from the edges. Improvements such as spatial coherence and multi-scale consistency are *explored*, not yet done.
6. **Mask and evaluation:** merge the chosen clusters into a binary mask, upsample it to the original image size, then compare with the ground truth.

The summary box repeats the key point: ground truth is used only for evaluation, never during inference.

**Speaker notes:**
"Here is the pipeline in six steps. One: resize and normalise the image. Two: pass it through a frozen DINOv2 ViT-B/14. It cuts the image into 14-by-14 pixel patches, so a 588 by 448 image gives 1,344 patches, and each patch is described by 768 numbers. Three: I use PCA to compress those to three numbers only to draw pictures. For the actual clustering I keep all 768, so no information is lost. Four: K-means groups similar patches, and I study how the number of clusters matters. Five is the key step: deciding which clusters are the object. The baseline is the border heuristic. Water and background usually touch the image edges, so clusters that touch the edges least are called foreground. Six: I merge those clusters, enlarge the result back to the original size, and score it against the ground truth. The ground truth never enters the process before this last step."

**Likely questions:**
- "Why DINOv2 and not CLIP or SAM?" Answer: DINOv2's patch features are known to be good for dense tasks without labels. Other backbones could be a later comparison.
- "Why K-means?" Answer: it is simple, fast and needs no training. Better clustering is possible later.
- "What if the object touches the border?" Answer: this is a known failure of the border rule. I will study it in the failure analysis, and better selection rules are part of the plan.
- "Why 14×14?" Answer: it is the patch size of the ViT-B/14 architecture.

---

## Slide 7: Evaluation strategy and performance metrics

**What it says:** Three metrics (IoU, F-measure, MAE), a qualitative analysis plan, a timing plan, and a summary box. (Remove the leftover sentence, see fix 1.)

**Explanation:**
- **IoU** = |P ∩ G| / |P ∪ G|. P is the predicted foreground and G is the ground truth. It is 1 for a perfect overlap and 0 for none.
- **F-measure** combines precision and recall. With β² = 0.3 (the standard salient-object setting), **precision counts more than recall**.
- **MAE** is the average per-pixel absolute difference between prediction and ground truth. Lower is better.
- *Qualitative analysis:* look at hard cases such as small objects, busy reefs, haze, objects at the image border, and foregrounds that look like the background.
- *Speed:* processing time per image and FPS, which matters for a robot.

**Speaker notes:**
"I evaluate with three standard binary-segmentation metrics. IoU is the overlap between my mask and the ground truth divided by their union, so higher is better. The F-measure balances precision and recall, with weight 0.3 on recall, so it punishes false foreground more than missed foreground. MAE is the average pixel error, lower is better. Numbers alone can hide problems, so I will also look at difficult images: small objects, complex reefs, haze, and objects touching the image border. Finally I measure time per image, because for an AUV speed matters. I will report only what I measure, and I will keep the oracle results clearly separate from the label-free method."

**Likely questions:**
- "Why β² = 0.3?" Answer: it is the convention in salient-object detection work, which emphasises precision.
- "What is a good IoU?" Answer: it depends on the dataset. Compare against baselines rather than a fixed number, and note that clustering without training will be below trained models.
- "What are your baselines?" Answer: simple PCA and 2-cluster K-means first. The SVAM-Net model (whose checkpoints come with the dataset) is a planned comparison.

---

## Slide 8: Expected outcomes and deliverables

**What it says:** Five deliverables, and a binary mask image (see fix 4).

**Explanation:**
1. A reproducible implementation for extracting frozen DINOv2 features.
2. A training-free segmentation pipeline with a label-free foreground rule.
3. Quantitative evaluation (IoU, F-measure, MAE) plus visual comparisons.
4. A study of what affects performance: number of clusters, which features are clustered, object size, background complexity, and the foreground-selection rule.
5. An assessment of computational cost and the practical limits of the method.

**Speaker notes:**
"By the end of the project I expect five deliverables: a reproducible feature-extraction and segmentation pipeline; a training-free method with a label-free foreground rule; a quantitative and visual evaluation; a study of which factors affect quality, such as the number of clusters, which features are clustered, object size, background complexity and the selection rule; and an honest account of the computational cost and limitations. The image on the right is an example of the kind of binary mask the pipeline outputs. I expect the method to work well on clean scenes and to struggle with cluttered reefs and objects at the image edge, but I will report what the measurements show."

(Delete the sentence about the image if you remove the picture. Do not promise numbers.)

**Likely questions:** "Will it be real-time?" Answer: the backbone itself is fast. The bottleneck so far is clustering on the CPU. Real-time on an embedded GPU has not been tested.

---

## Slide 9: Thank you

**Speaker notes:**
"Thank you. To recap, the project asks whether frozen DINOv2 features and unsupervised clustering can separate objects from the background in underwater images without training. I have defined the task, the dataset, the method and the evaluation. I'm happy to take questions."

---

## Backup: optional facts if asked about early results

Say these only if asked, because Review 1 is about the plan. All numbers are means over the 300 USOD images and come from the CSV files in `results/usod/`.

- Simple baselines (2 clusters, border rule): PCA mask IoU 0.418, K-means IoU 0.490.
- With the oracle choosing the best cluster combination using ground truth (an upper bound): IoU about 0.66–0.70.
- With the label-free border rule: IoU about 0.45–0.56. The gap between the two shows that **picking the foreground cluster is the weak step**, not the features.
- Using the full 768-D features for clustering beat using the 3-D PCA colours.
- Running DINOv2 at a finer stride (dense features) made no meaningful difference to IoU and was far too slow.
- Measured on an RTX A2000, not a Jetson: the backbone takes about 17 ms per image at 448×588, but the full label-free pipeline runs at about 2.5–6.5 FPS because of CPU clustering.
- Not yet done: SUIM, an SVAM-Net baseline, and tests on empty-water images (every USOD image contains an object, so false alarms are unmeasured).

---
---

# Part 2: Concept explainers (for answering questions)

These sections explain the ideas behind slides 2, 5, 6 and 7 in more depth. All numbers and settings are the ones used in the code (`src/dinov2_seg.py`, `src/dinov2_cluster_union.py`).

## A. What DINOv2 does

**One-line version:** DINOv2 turns an image into a grid of numbers, 768 for each small square (patch) of the image, such that squares that look or behave alike get similar numbers.

**Step by step (ViT-B/14, as we use it):**
1. **Cut into patches.** The resized image (448 tall × 588 wide) is cut into 14×14-pixel squares: 32 rows × 42 columns = **1,344 patches**.
2. **Embed each patch.** Each square's 14×14×3 pixel values are turned into a 768-number vector by one linear layer. A position code is added so the model knows where each patch sits. An extra "CLS" token is added that summarises the whole image. We do not use it.
3. **Transformer layers (12 of them).** In each layer every patch looks at all the other patches (*self-attention*) and updates its vector using what it finds relevant. After 12 rounds, each patch's vector contains information about the patch itself and its context in the image.
4. **Output.** We keep the 1,344 patch vectors (1,344 × 768) and ignore the CLS token.

**Why the output is useful:** DINOv2 was trained, without any labels, on about 142 million curated images. It learned by making the features of different views of the same image agree. To do that it had to learn what is *in* the image. As a result, patches from the same kind of thing (fur, water, coral, a fish body) end up with similar vectors even though the model never saw a label such as "fish". We use that property and never train or change the model, which is what "frozen" means.

**What it does not do:** it does not output a mask or a class name. It only gives features. Everything after that (PCA, k-means, picking the foreground) is our own simple processing on those features.

**Caveat to state if asked:** DINOv2 was trained on everyday images, not underwater images. Whether its features still separate underwater objects well is exactly what the project tests.

---

## B. What PCA does, and how it becomes colours (RGB)

**The problem:** each patch has 768 numbers. We cannot draw or look at 768 dimensions.

**What PCA (Principal Component Analysis) does:**
1. Put all 1,344 patch vectors into a table (1,344 rows × 768 columns).
2. Subtract the average vector, so the cloud of points is centred.
3. Find the direction in 768-D space along which the patches **vary the most**. This is **PC1** (the first principal component). Then find the next most varied direction at right angles to it (**PC2**), then the next (**PC3**).
4. Describe each patch by how far it lies along those three directions. Every patch now has **3 numbers instead of 768**, and those 3 numbers keep as much of the variation as 3 numbers can.

**How the 3 numbers become a colour:**
- PC1 → **red**, PC2 → **green**, PC3 → **blue**.
- Each channel is rescaled to run from 0 to 1 over this image: (value − smallest) / (largest − smallest).
- The 32×42 colour grid is shown as an image. This is the right-hand picture on slide 5.

**How to read it:** patches with similar colours have similar features. A region with one flat colour is something DINOv2 sees as "one kind of thing". Colours **change from image to image**: PCA is fitted separately on each image, and the sign and order of the components are arbitrary. So pink on one image has no fixed meaning (for example, it does not mean "fish").

**What PCA is used for in this project:**
- **Visualisation:** the colour picture.
- **Baseline mask (E1):** take PC1 > 0 as one group and the rest as the other, then decide which side is the foreground with the border rule.
- **Compared with using all 768 numbers for clustering.** We found clustering on the full 768 numbers worked better than clustering on the 3 PCA colours, because the 3 colours throw information away. That is why slide 6 says "retain original features for clustering".

---

## C. What k-means does

**The goal:** group the 1,344 patches into *k* groups so that patches in a group have similar features.

**The algorithm (what happens inside):**
1. Pick *k* starting centres.
2. **Assign:** give each patch to the nearest centre (straight-line distance between the 768-number vectors).
3. **Update:** move each centre to the average of the patches assigned to it.
4. Repeat steps 2 and 3 until nothing changes.

**Our settings:** we run this separately on each image, with 4 different random starts (`n_init=4`) and keep the best, and a fixed seed so results repeat. Each patch ends up with a **cluster label** (0, 1, …, k−1). Reshaped into the 32×42 grid, the labels form a coarse segmentation of the image.

**Turning labels into a mask:** pick which clusters count as foreground, mark their patches as 1 and all others as 0, then enlarge the 32×42 grid to the original image size with bilinear interpolation and threshold at 0.5 (this is the "upsample" in step 06 on slide 6).

**Two important limits:**
- **The labels have no meaning.** "Cluster 0" is not "water". K-means has no idea which group is the object, so a separate rule is needed. That is what the border rule and the oracle are about (sections E and F).
- **The number *k* must be chosen.** Too few clusters merges object and background. Too many splits the object into pieces (for example a fish's body, fins and eye). That is why we use the silhouette score (section D).

---

## D. Silhouette score

**What it answers:** "How good is this clustering?" We use it to choose *k* for each image without ground truth.

**How it is computed, for each patch:**
- *a* = the average distance from this patch to the other patches **in its own cluster** (small is good: tight cluster).
- *b* = the average distance from this patch to the patches in the **nearest other cluster** (large is good: well separated).
- Silhouette of the patch: **s = (b − a) / max(a, b)**.

**How to read it:**
- Close to **+1**: the patch sits snugly inside its cluster and far from the others.
- Around **0**: it is on the border between two clusters.
- Negative: it is probably in the wrong cluster.

The image's score is the average over all patches. We run k-means for each *k* (2 to 4, or up to 6 in later runs) and pick the *k* with the **highest** score. Call it **k\***.

**What to say about its limits:** it measures how well separated the groups are in feature space. It does **not** know about objects. A high score can mean "water vs. everything else" even when the object itself is split into two parts. It is a reasonable label-free guide, not a guarantee. The average chosen k\* was about 4 in the later runs.

---

## E. Border rule (the label-free foreground selector)

**The idea:** in most photos the object of interest is not spread along the image edges, while water, sea floor and background are. So a cluster that **touches the image border a lot** is probably background, and one that **mostly stays away** from the border is probably foreground.

**Exactly how we compute it:**
1. Take the 32×42 label grid and mark the **outer ring** of patches (the first and last rows and columns). That ring is 144 of the 1,344 patches, about **10.7 %** of the image.
2. For each cluster, compute its **border fraction**: the share of that cluster's patches that lie on the ring.
3. A cluster is **foreground** if its border fraction is **below the image's overall ring share (10.7 %)**. Otherwise it is background.
4. The foreground is the **union** of all clusters that pass. If none or all of them pass, fall back to the single cluster with the smallest border fraction.

**Why it is useful:** needs no labels, no training, and costs almost nothing.

**When it fails (the honest part):**
- **The object touches the border.** In `usod_test174` the reef reaches the image edge, so the rule treats it as background (clustering on the PCA colours: IoU 0.46 with the rule versus 0.78 for the best choice).
- **Clusters are right but the rule picks wrongly.** In `usod_test205` the fish is cleanly clustered (best choice IoU about 0.81 on PCA colours), but the rule picks something else (about 0.16).
- **Large close-up objects** that fill the frame, or background that does not reach the edges.

---

## F. Oracle

**What it is:** a way to ask "if the clusters were chosen perfectly, how good could the mask be?"

**How it works:** for an image with *k* clusters there are 2^k − 2 ways to pick a non-empty, non-total set of clusters as the foreground (for k = 4 that is 14 ways). For each way, build the mask and compute IoU with the **ground truth**. Keep the best one.

**What it tells us:**
- It is an **upper bound** on what the clusters can do. In our data the oracle reached IoU about 0.66–0.70, versus about 0.45–0.56 for the border rule.
- The **gap between the two** shows that the features and clusters are decent, but **choosing the foreground without labels is the weak step**. This guides future work (a better selector).

**Why it must not be reported as the method's result:** it uses the ground truth, so a deployed AUV could never do it. It also gets **easier to inflate as k grows** (more subsets to choose from), so compare it only at the same k. Always label it "oracle (upper bound)", never as "our accuracy". This is the point made in the box at the bottom of slide 4.

---

## G. Performance metrics, exactly

Let **P** be the set of pixels the method calls foreground (after thresholding the upsampled mask at 0.5) and **G** be the ground-truth foreground pixels.

Count four things: **TP** = pixels in both P and G; **FP** = in P only (false alarm); **FN** = in G only (missed); the rest are true negatives.

### IoU (Intersection over Union)
- **IoU = |P ∩ G| / |P ∪ G| = TP / (TP + FP + FN)**
- Overlap divided by total area covered by either. **1** is perfect, **0** is no overlap.
- It punishes both false alarms and misses. It ignores the large background area, so it is not inflated by easy background.

### F-measure (weighted, β² = 0.3)
- **Precision = TP / (TP + FP)** = of the pixels I called foreground, how many were right.
- **Recall = TP / (TP + FN)** = of the true foreground pixels, how many did I find.
- **F = (1 + β²) · Precision · Recall / (β² · Precision + Recall)**, with β² = 0.3, so **F = 1.3 · P · R / (0.3 · P + R)**.
- β² = 0.3 is the standard choice in salient-object detection. It weights **precision more than recall**, so a mask with a lot of false foreground is penalised more than one that is slightly too small.

### MAE (Mean Absolute Error)
- **MAE = average over all pixels of |prediction − ground truth|**, with values between 0 and 1.
- It is the fraction of pixels that disagree. **Lower is better**, 0 is perfect.
- In our code the prediction used here is the **upsampled soft mask** (values between 0 and 1 near edges, before thresholding), so MAE is a little smoother than a hard pixel count.
- Unlike IoU, it counts the background too, so for small objects a mask that says "nothing" can still get a low MAE. That is why we never read MAE alone.

### Worked example (made-up numbers, to explain the formulas)
A 1,000-pixel image. Ground truth has 100 foreground pixels. The method marks 120 pixels, of which 80 are correct.

| Quantity | Value |
|---|---|
| TP / FP / FN | 80 / 40 / 20 |
| IoU = 80 / (80 + 40 + 20) | **0.571** |
| Precision = 80 / 120 | 0.667 |
| Recall = 80 / 100 | 0.800 |
| F = 1.3 · 0.667 · 0.8 / (0.3 · 0.667 + 0.8) | **0.693** |
| MAE = (40 + 20) / 1000 (hard mask) | **0.060** |

### How we report them
Each metric is computed per image and then **averaged over the 300 USOD images**. The medians and per-image CSV files are also available. IoU is the headline number. F-measure and MAE are the usual salient-object companions.

**A one-sentence answer if a reviewer asks "why all three?":** IoU measures overlap, F-measure shows whether errors are mostly false alarms or misses (with precision weighted higher), and MAE measures overall per-pixel disagreement, so together they cover different failure types.


---
---

# Review 2 deck: speaker notes (slides 3-16)

File: `Review-2-Training-Free-Underwater-Object-Segmentation.pptx` (16 slides). Slides 1, 2 and 16 come from your own deck: slide 1 now says PROJECT REVIEW 2 and has four lines for names and roll numbers; slide 2 is unchanged. (The notes above this line belong to the old Review 1 deck.)

**How to use these notes.** Each slide has three parts: **Point at** (what to show with your hand or the pointer, in order), **Say** (simple words you can read out), and **If asked** (short answers). The "Say" parts are short on purpose, about one minute each. Say them in your own words.

**Words used in the talk (keep them simple):**
- **patch** = a small 14 × 14 pixel square of the photo. **group** = patches that look alike (also called a cluster).
- **our mask** = the object mask we get with **no labels**. **true mask** = the mask drawn by people. **best choice** = the best possible mask from the same groups, picked **using** the true mask (an upper bound, not a method).
- **IoU** = how much two masks overlap, from 0 (no overlap) to 1 (the same). **Mean** = average over the 300 images.

**Time guide for 15 minutes:** slide 1: 0.5 min, 2: 0.75, 3 to 11: 1 min each, 12: 1.25, 13: 0.75, 14: 1.25, 15: 1.5. If you run late, skip slide 13.

**Before presenting, fill in:** the four `[Student name] · [Roll number]` lines and `[Project guide]` on slide 1, and `[date]` for Review 3 on slide 15. The deck uses the fonts Arimo and M Plus 1m; if the presenting computer does not have them, open the file once there and check that no text runs outside its box.

---

## Slide 3: Why it matters, and the cost of ignoring it

**Point at:** (1) the three rows on the left, top to bottom: "Labels are scarce", "Water hides the object", "Vehicles need a cue"; (2) the two photos on the right; (3) the blue box at the bottom left.

**Say:** "Why does this matter? Look at the three reasons on the left. First, labels are scarce: someone has to draw every object by hand, and our dataset has only 300 images. Second, water hides the object: light fades, the picture gets hazy and the colours change, so simple colour rules do not work. Third, a robot with a camera, like the vehicle in the top photo, must know that something is in front of it, so it can avoid it, inspect it or follow it. Now the blue box: if we do not solve this, every new place or camera needs a new set of hand-labelled images, or the robot stays blind to what is ahead. The robot is our motivation. In this project we only test the picture-cutting part."

**If asked:** *Did you test it on a vehicle?* No. We only measure masks on images. *Why not train a network?* That needs many labelled underwater images, which are rare. We ask how far we get with no training.

---

## Slide 4: How the method works, step by step

**Point at:** the five pictures from left to right, then the blue box.

**Say:** "Here is the whole method in five pictures. One: the photo goes in. Two: we cut it into small squares, 14 by 14 pixels. That gives 1,344 patches, the grid you see. Three: a ready-made model, called DINOv2, gives each patch 768 numbers, a kind of fingerprint. We cannot draw 768 numbers, so this picture squeezes them into three colours. Patches with the same colour look alike: the diver is pink, the reef is green and blue, and the water is pale yellow. Four: we put patches that look alike into the same group, shown here in different colours. Five: we pick the groups that are the object. That is our mask. The blue box says it in one line: it is like sorting jigsaw pieces by how they look, and then deciding which piles are the fish and which are the water. Nothing is trained; the model is used as it is."

**If asked:** *What is DINOv2?* A model from Meta, trained on millions of images without labels. We do not change it. *Why 768?* It is the size of the model's description for each patch. *What is PCA?* A way to squeeze many numbers into three, only to draw the colour picture.

---

## Slide 5: What we measure: overlap with the true mask

**Point at:** the four pictures from left to right (photo, true mask, our mask, overlap), then the box on the left, then "300" on the right.

**Say:** "To compare methods we need one number. Look at the pictures. This is the photo. This is the true mask drawn by people. This is our mask. And the last picture puts them on top of each other: green is where we are right, red is where we said object but it is water, and yellow is what we missed. Now the box. IoU means overlap divided by the area that either mask covers. Here it is 31 thousand divided by 57 thousand, so 0.54. One means a perfect match, zero means no overlap. Every number you will see later is the average IoU over these 300 images."

**If asked:** *Why not accuracy?* Most pixels are water, so saying "all water" would look accurate. IoU looks only at the object. *Is the true mask the truth?* It is what the people marked as the main object.

---

## Slide 6: PCA-RGB maps: stride 14, then stride 3

**Point at:** the three columns (photo, stride 14, stride 3), going down the three rows; then the text on the right; then the blue box.

**Say:** "These are the colour maps from slide 4, for three images. In the middle column the map is blocky, because each square is looked at separately. That is stride 14. Now look at the right column: the map is much smoother. We got that by sliding the squares in small steps of 3 pixels instead of 14. In both maps the colours separate the diver, the fish and the water, without any training. But read the blue box: when we scored it, the result hardly changed, 0.703 against 0.702, and it takes about 17 times longer. So a nicer picture did not mean a better mask. That was our first lesson."

**If asked:** *What is stride?* The step between the places where the model looks. *Why keep it then?* Better pictures. We saw later it does not help the score either.

---

## Slide 7: k-means with up to 4 groups

**Point at:** the top row, left to right: photo, true mask, groups (k up to 4), best choice, our mask; then the same for the second row; then the box at the bottom.

**Say:** "Now the first clustering method, k-means. Look at the fish in the top row. Column three shows the groups k-means found, with at most 4 groups. The fish is spread over several groups. Column four is the best mask we could make from these groups, if we pick them using the true mask. It is speckled, only 0.23. Our real mask, with no labels, in the last column, is even worse, 0.10. The diver in the second row is the same story: 0.31. The box gives the averages over 300 images: 0.676 for the best choice and 0.560 without labels. We chose these two images because they show clearly what happens next."

**If asked:** *What is the rule that picks the groups?* The border rule: a group is the object if fewer of its patches touch the image border than expected, because water touches the border and the object usually does not. *Why k-means?* It is simple, standard and needs no training.

---

## Slide 8: Same images with up to 6 groups

**Point at:** the two rows (the fish, then the diver). Read the columns left to right: the photo; then the block "k up to 4" (groups, best choice, our mask); then the block "k up to 6" (groups, best choice, our mask). Finish with the box at the bottom.

**Say:** "These are the same two images as the last slide, now with the two settings side by side. Start with the fish. In the 4-group block, the groups are rough, the best choice is speckled, 0.23, and our mask, with no labels, is 0.10. Now the 6-group block. The groups are finer, the fish has its own light-blue group, and the best choice is a clean fish, 0.67. So the clustering really got better. But look at the last column, our mask with no labels: 0.21, still bad. The diver is the same: the best choice goes from 0.31 to 0.71, our mask only from 0.31 to 0.36. The box says it in numbers: the best case rises on average from 0.68 to 0.70, but our own score goes down, from 0.56 to 0.53. So more groups give us a better mask to choose from, but we cannot choose it without labels."

**If asked:** *Why does our mask not improve?* With more groups there are more ways to choose wrongly, and the border rule cannot tell which groups form the object. *What is the best choice?* The best mask from the same groups, picked using the true mask. It shows what is possible; it is not a method. *Is this typical?* The best case improves by more than 0.10 on 29 of 300 images, so these two are clear examples, not the average.

---

## Slide 9: The groups are good; choosing them is hard

**Point at:** the two rows on the left (photo, best choice, our choice); then the bar chart; then the blue box.

**Say:** "Here are two more images. Look at the middle column: the best choice of groups gives a very good mask, 0.96 and 0.94. The right column is what we get without labels: 0.75 and 0.69. Still decent, but clearly worse. Now the chart: across all 300 images the best choice is 0.70 and ours is 0.52, in the first pair of bars. That gap is the problem. The groups already contain a good mask. What we are bad at is deciding which groups are the object. So next we try a different kind of grouping that tells us more: a dendrogram."

**If asked:** *Is 0.70 a fair result to quote?* No. It uses the true mask, so it is an upper bound. It only shows what is possible from these groups.

---

## Slide 10: A dendrogram is a family tree of patches

**Point at:** (1) the small tree on the left, from the bottom up; the red dashed line and the green line; (2) the three numbered lines under the tree; (3) the four pictures on the right.

**Say:** "A dendrogram is a family tree. Look at the small example on the left. At the bottom each piece starts alone: one fish, two reef, three water. We join the two closest pieces, again and again. The height of a joint tells you how different the two pieces were: the water pieces join low because they are alike, and the fish joins last, very high, because it is very different. Now a cut. If we cut along the red dashed line, we get two groups: fish, and everything else. If we cut along the green line, lower, we get three groups: fish, reef, water. So one tree gives every number of groups. Now the pictures on the right: this is a real fish image. The same tree, cut at three heights, gives 2 groups, 3 groups and 4 groups, and each time one group is split in two. The tree also tells us how different the groups are, and we use that next."

**If asked:** *How do you choose where to cut?* Two ways, shown later: the silhouette score, or the "gap", the biggest jump in height. *Why only a few branches in the real trees?* The pictures of trees show only the top branches, the full tree has every patch.

---

## Slide 11: Four ways to measure "closest" between groups

**Point at:** the top row (photo, true mask, k-means 0.85); then the bottom row from left to right: Ward, Average, Complete, Single (the white one); then the box.

**Say:** "When we join groups, we need a rule for how far apart two groups are. That rule is called linkage, and there are four common ones. All four in the bottom row use the same photo and the same way of choosing the object. Ward and Average give a clean fish, 0.87 and 0.83. Complete is already worse, 0.56. Now the last one, Single: the whole picture is white. Here is how that happens, in three steps. One: Single linkage joins groups through chains of close neighbours, so almost every patch ends up in one huge group, 99.7 percent of them. Two: the few patches left out, 81 of them, sit mostly on the border of the picture, so our border rule calls them water. Three: the huge group is called the object, so the mask is all white. A white picture covers the fish and everything else, and the fish is only 18 percent of the picture, so the score is 0.18. That is no better than saying everything is object. It was one of our failed experiments. On average over 300 images Single gets 0.22, and Average works best, 0.65."

**If asked:** *What does "all marked object" mean?* The method called every patch of the picture the object, so the mask is all white. *Why does Single fail?* It joins two groups as soon as their closest patches are close, so long chains swallow everything. *Did you try other tree types?* Yes, trees where only neighbouring patches may join. Most failed too, only Ward stayed good.

---

## Slide 12: Dendrogram versus k-means on USOD

**Point at:** the bars from left to right; the green bar; the yellow bar; then the three boxes on the right from top to bottom.

**Say:** "This is the main result. Each bar is the average IoU over 300 images, with no labels used. The grey bar is k-means: 0.52. The blue bars are dendrograms: Ward and Complete 0.57, Average 0.63. The green bar is Average with the gap cut, which means we cut the tree at the biggest jump in height: 0.65. The yellow bar is the best possible from the same groups, 0.71. So we closed about two thirds of the gap. On the right: the gain over k-means is 0.12, and the interval says it is certainly above zero. The middle number says that the picture is better for 52 percent of the images and clearly worse for 20 percent. So it is better on average, not on every image. The last box: on a second dataset, SUIM, with 108 images, we see the same gain, 0.12."

**If asked:** *Is the gain real?* The interval stays well above zero and SUIM agrees. But we chose the best method on the same images we score it on, so a test on new images is part of our plan. *What is the gap cut?* Cut the tree where the jump in height is largest. *Why Average?* It was the best of the four in our tests.

---

## Slide 13: Where the dendrogram fixes k-means mistakes

**Point at:** the three rows; in each row the third picture (k-means), then the fourth (dendrogram); then the box on the right.

**Say:** "Here are three images where k-means gets it wrong. Look at the third column: the mask is speckled all over the picture, IoU 0.11, 0.07 and 0.10. K-means picked the wrong groups. In the fourth column the dendrogram finds the object: 0.84, 0.78 and 0.72. I chose these because they show the effect clearly. They are among the biggest gains, not the typical case. The typical case is on slide 12: better on about half of the images."

**If asked:** *Are these cherry-picked?* They are among the largest gains, chosen to show the effect. Slide 12 gives the honest summary and the next slide shows a failure.

---

## Slide 14: When it fails: the same method on usod_test40

**Point at:** the top row: photo, true mask, gap cut 2 groups (almost all white), silhouette 4 groups; then the tree on the bottom left, with the two lines and "only 6 patches"; then the text on the right.

**Say:** "Now the other side. This is the same method on a different image. Look at the third picture: the mask is almost all white, IoU 0.21. The fourth picture is the same method with a different cut, and it is good, 0.83. So what happened? Look at the tree. The red dashed line is the gap cut: it cuts at the top and makes 2 groups. But the left branch holds only 6 patches out of nearly 28 thousand, a few odd patches. The gap rule thinks that is the biggest jump, so it cuts there. Then everything else is called the object, so almost the whole picture is white. The green line, which makes 4 groups, works. This happens in about 1 image out of 5. Our next step is to stop tiny groups from winning the gap rule. One note: the picture of the tree shows only the top 12 branches. The full tree contains all the patches."

**If asked:** *Is that really the reason?* The group sizes, 6 patches against 27,834, and the low score at 2 groups are measured. Why the gap is largest there is our explanation. *Why not always use the silhouette?* It was slightly worse on average, 0.63 against 0.65.

---

## Slide 15: Conclusion and plan for Review 3

**Point at:** the three big numbers at the top; the "Limits" list on the left; the plan on the right.

**Say:** "To sum up. The three boxes: without labels, the score went from 0.52 with k-means to 0.65 with a dendrogram. The best possible from the same groups is 0.71, so the groups were good already, and the gain comes from choosing better. And it is better for 52 percent of images but worse for 20 percent. Now the limits on the left. We picked the best method on the same images we score it on. On a second kind of label, everything except water, saying everything is object already scores 0.52, so those numbers tell us little. We only tried 2 to 6 groups, and the rule settings are not tuned. And we have not tested the speed on a vehicle. On the right is our plan for Review 3, the final review. One: fix the failure we just saw and test on new images. Two: compare with a trained method and run ablations. Three: test on more data. Four: measure the speed for a vehicle, and write the final report."

**If asked:** *What will you do differently by Review 3?* The four items on the right. *Will it run on the vehicle?* Not shown yet. Earlier we measured about 2.5 to 6.5 pictures per second on a desktop graphics card for the simple version. *Does it beat trained methods?* We do not know yet. The comparison is in the plan.

---

## Slide 16: Thank you

**Say:** "To close: the groups were already good, and a dendrogram helps us choose the object without labels, and we know where it fails. Thank you. We are happy to take your questions."

---

## Questions you may get on any slide (short answers)

- **Why no training?** Labels are scarce and costly. We test how far a ready-made model goes.
- **What is new here?** We found that the groups are good, but choosing the object is the weak link. A dendrogram with a gap cut improves the choice without labels: +0.12 on 300 images, and the same gain on a second dataset.
- **What failed?** Smoother maps did not help the score. More k-means groups did not help our choice. Single linkage failed. The gap rule fails when a tiny group splits off first.
- **How do you know it is not luck?** We compared image by image and the gain stays well above zero. But we picked the method on the same images, so a test on new images is planned.
- **Which data?** USOD, 300 images with one main object each; SUIM test set as a second check.
- **How fast is it?** About 1.5 seconds per image for all the tree variants together at stride 14 on a desktop graphics card. The simple version ran at about 2.5 to 6.5 pictures per second. A test on the vehicle is not done.
