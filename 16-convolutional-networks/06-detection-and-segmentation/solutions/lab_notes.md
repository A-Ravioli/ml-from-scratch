# detection & segmentation · lab notes

all three experiments generate their own synthetic data with fixed seeds, so the numbers are the same offline and online. measured with the reference solution on an Apple M2 laptop CPU (under load from other jobs; the whole lab took about 80 s). experiments 1 and 2 are robust to the seeds. experiment 3 is not, in its details: a reviewer re-ran it with six seed offsets, and Dice's IoU ranged from 0.67 to 0.85, and whether Dice alone or BCE + Dice finds the most tiny objects changed from seed to seed (Dice did in 3 of 6). what held in every run: BCE has the best (or tied best) IoU, BCE + Dice finds more tiny objects than BCE alone, and Dice alone has the worst IoU. the notes below rely only on those.

## experiment 1: choosing the NMS threshold

**what people usually predict:** there's a sweet spot around the default 0.5 that works everywhere.

**what happens:** mean precision and recall over 200 scenes (after dropping boxes scoring below 0.45):

| threshold | sparse precision | sparse recall | crowded precision | crowded recall |
|---|---|---|---|---|
| 0.3 | 1.000 | 0.955 | 1.000 | 0.498 |
| 0.4 | 1.000 | 0.978 | 1.000 | 0.568 |
| 0.5 | 0.991 | 0.993 | 0.994 | 0.780 |
| 0.6 | 0.876 | 0.998 | 0.900 | 0.947 |
| 0.7 | 0.560 | 1.000 | 0.589 | 0.996 |
| 0.8 | 0.322 | 1.000 | 0.333 | 1.000 |

(0.1 and 0.2 lose even more recall, 0.9 even more precision.) in sparse scenes the default 0.5 is nearly perfect (0.99 / 0.99). in crowded scenes no threshold gets recall above 0.95 with precision above 0.9: at 0.5, NMS deletes 22% of the objects; at 0.6 recall is 0.947 and precision 0.900; at 0.7, 41% of the reported boxes are duplicates.

**why:** greedy NMS sees only IoUs, and in a crowd the two kinds of overlap it must distinguish have the same range. a neighbour shifted by 15–30% of its width has IoU $\frac{1 - \delta}{1 + \delta} = 0.54$–$0.74$ with its partner (on paper 1); a duplicate detection jittered by about 7% of the box size per coordinate has IoU between 0.63 and 0.85 (for 80% of pairs) with another detection of the same object. a low threshold suppresses the neighbour together with the duplicates (recall falls), a high one keeps the duplicates together with the neighbour (precision falls); at low thresholds even sparse scenes lose objects that happen to overlap by chance. no threshold separates two overlapping distributions. remedies change the question rather than the threshold: soft-NMS decays overlapping scores instead of deleting them (Bodla et al., 2017), learned NMS and crowd-specific heads predict whether two boxes are the same object, and DETR (Carion et al., 2020) removes NMS entirely by predicting a set of boxes with a one-to-one matching loss, so the network itself learns not to duplicate.

## experiment 2: which objects do the anchors cover?

**what people usually predict:** objects of every size get a decent match; small ones are a bit worse.

**what happens:** best anchor IoU for 3,000 random boxes, by size:

| object size | objects | median best IoU | covered at IoU ≥ 0.5 |
|---|---|---|---|
| 4–8 | 548 | 0.03 | 0.0% |
| 8–16 | 533 | 0.12 | 0.0% |
| 16–32 | 546 | 0.44 | 35.2% |
| 32–64 | 511 | 0.60 | 86.5% |
| 64–128 | 525 | 0.66 | 99.8% |
| 128–200 | 337 | 0.62 | 79.2% |

objects under 16 pixels are *never* matched at IoU 0.5, and below 8 pixels the best IoU is 0.03. coverage peaks between the anchor sizes and dips both between them (the saw-tooth in the plot, at sizes $32\sqrt 2$ and $64\sqrt 2$) and above the largest. for a scene with 50 objects, `match_anchors` labels 202 of 2,304 anchors positive, 265 ignored and 1,837 background.

**why:** a small object centred in an anchor of side $a$ has IoU $(b/a)^2$ at best (on paper 2): with $a = 32$, a 16-pixel object reaches 0.25 and an 8-pixel one 0.06. no amount of training can make such an object a regular positive: its only positive example per image is the one anchor `match_anchors` forces, with a regression target that has to shrink the anchor fourfold. so small objects get few, hard training examples, and their detections are poor. feature pyramids fix the geometry by adding a finer map (stride 4 or 8) with 8- or 16-pixel anchors, and more scales per octave fill the saw-tooth dips. the label counts show the other structural problem: background anchors outnumber positives about 9 to 1 here, and 1,000 to 1 in a real image with a few objects and 100,000 anchors. with plain cross-entropy the easy negatives dominate the loss, which is what RetinaNet's focal loss (Lin et al., 2017) down-weights.

## experiment 3: segmenting small shapes

**what people usually predict:** Dice wins everything on imbalanced data; or cross-entropy wins because it's "standard".

**what happens:** objects cover 3.7% of the pixels:

| loss | test IoU | small objects found | predicted foreground fraction |
|---|---|---|---|
| BCE | **0.899** | 85.3% | 0.035 |
| dice | 0.685 | **92.7%** | 0.049 |
| BCE + dice | 0.889 | 91.3% | 0.036 |

cross-entropy gives the best overall IoU but misses 15% of the tiny objects (radius below 3); Dice alone gives much worse masks (IoU 0.69 here, and the worst IoU in every seed), with a third more foreground predicted than there is; the sum keeps nearly all of BCE's IoU and finds more of the tiny objects than BCE does (91% vs 85% here, and more than BCE in every seed). in this run Dice alone found the most tiny objects, but that ranking between Dice and the sum flips with the seed.

**why:** each loss has its own bias. cross-entropy averages over pixels, so a missed 9-pixel object costs about as much as 9 wrong background pixels out of 1,024: little. under uncertainty its optimum is calibrated per pixel, which for faint, tiny objects means probabilities below 0.5 and no detection; but where it does predict, its masks are tight (predicted foreground 0.035 against a true 0.037). Dice normalizes by the size of the object, so a missed tiny object costs as much as a missed large one, and it pushes towards finding everything (how much of that recall survives its noisier optimization varies from run to run); but its gradient for a background pixel is small once the object is found, and it rewards overlap, not calibration, so it happily grows blobs a pixel or two beyond the boundary (and occasionally hallucinates a speck): foreground 0.049, IoU 0.69. the sum gets the per-pixel calibration from cross-entropy and the object-level pressure from Dice, which is why "BCE + Dice" is the common default in medical imaging, where lesions are small and every one counts.
