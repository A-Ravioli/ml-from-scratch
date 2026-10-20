# multimodal learning · lab notes

numbers are from the reference solution with the lab's defaults (5 models, seeds 0–4, trained on an Apple-silicon GPU via MPS; 153 seconds in all on a lightly loaded machine). training on MPS is not bit-for-bit reproducible: earlier runs of this lab (with 3 models) and other runs gave numbers that moved by several points, so only findings that hold on *every* model are stated as findings. yours will differ in detail.

a reminder of what the data is: real MNIST digits, recoloured and placed by a program, with captions generated from 6 templates over a 30-word vocabulary. it's a world where every caption is a clean, true combination of independent attributes. that makes the *mechanics* of CLIP visible on a laptop; it says little about what a model trained on real web captions would do.

## experiment 1: does the wording of the prompt matter?

**what people usually predict:** the more natural prompt ("a photo of a {d}") wins, as it does for real CLIP; numerals are "a bit worse"; the colour-specific prompt hurts.

**what happens:** zero-shot digit accuracy on 3,000 test images (409 of them red), mean over 5 models (min–max), and the paired difference from the bare-word prompt on the same model:

| prompt | accuracy | on red digits | Δ vs `{d}` | Δ on red digits | unknown words |
|---|---|---|---|---|---|
| `{d}` | 0.836 (0.792–0.885) | 0.744 (0.620–0.882) | – | – | 0 |
| `a {d}` | 0.840 (0.790–0.896) | 0.765 (0.690–0.905) | +0.004 (−0.016 to +0.017) | +0.021 (−0.042 to +0.113) | 0 |
| `a photo of a {d}` | 0.813 (0.772–0.848) | 0.758 (0.653–0.928) | −0.023 (−0.057 to +0.020) | +0.014 (−0.109 to +0.141) | 2 |
| `the digit {n}` (numerals) | 0.090 (every model) | 0.097 | −0.745 (−0.795 to −0.702) | −0.647 (−0.785 to −0.523) | 1 |
| `a red {d}` | 0.853 (0.800–0.905) | 0.878 (0.803–0.912) | +0.017 (−0.019 to +0.039) | +0.134 (+0.025 to +0.292) | 0 |
| ensemble of 28 `a {c} {d} in the {p}` | 0.860 (0.822–0.897) | 0.791 (0.713–0.917) | +0.024 (0.000 to +0.051) | +0.047 (−0.014 to +0.095) | 0 |

two findings hold on every model, in this run and in every earlier one: **numerals collapse to the class-0 base rate** (0.090), and **"a red {d}" helps red digits** (by 0.03 to 0.29 here; other runs saw +0.04 to +0.29). everything else is smaller than the variation between models or between runs: the ensemble's gain is never negative here but is at most 0.05; "a photo of a {d}" lowered overall accuracy on average (by 0.023, ranging from −0.057 to +0.020 across models; other runs saw down to −0.08), which suggests a small real cost but not a reliable size.

**why:**

- **numerals:** the training captions spell digits as words, so "0" … "9" are all `<unk>`. every class prompt becomes the identical token sequence "the digit `<unk>`", all ten class embeddings are equal, and the argmax always returns class 0 (which is 9% of the test set). the model isn't bad at the task; the prompt asks a question in a language it never learned. this is the cartoon version of a real effect: CLIP does well with prompts that look like web captions and poorly with phrasings that are rare on the web.
- **unknown words** ("photo", "of") push every class prompt in the same direction, so they mostly cancel in the argmax; what's left is a small, model-dependent distortion, usually for the worse. with real CLIP the same prompt *helps*, because "a photo of a …" is common in web captions; here it's two `<unk>`s.
- **`a red {d}`** makes the prompts look like the training captions of red digits, and red digits are where the bare-word prompt is weakest (0.74 against 0.85 for the other colours; we haven't pinned down why). for other colours, *every* class prompt gains the same word, so the ten embeddings move together and their ranking for a green image barely changes (−0.002 on average for non-red digits). a mismatched description doesn't hurt much when it's shared by all classes.
- **the ensemble** averages over every colour and position the captions mention, so each class embedding sits in the middle of everything a caption of that digit could say. it never hurt here, as Radford et al. found for real CLIP, but its gain is small because our captions vary little.

the paired Δ columns are what make even these conclusions possible: the between-model spread (0.79–0.89 for the bare word) is larger than most of the prompt effects.

## experiment 2: combinations never seen in training

**what people usually predict:** the model fails on green sevens and red threes, or at least gets them wrong much more often.

**what happens:** colour and digit classification of the held-out combinations (90 test images) and of everything else (2,910), mean over 5 models (min–max):

| test images | n | colour accuracy | digit accuracy | both right |
|---|---|---|---|---|
| held-out combinations | 90 | 0.987 (0.933–1.000) | 0.851 (0.611–0.978) | 0.838 (0.611–0.944) |
| all other combinations | 2,910 | 0.993 (0.984–1.000) | 0.860 (0.819–0.896) | 0.854 (0.804–0.896) |

the robust finding is the colour: held-out images get it right on every model (93–100%). the digit varies widely between models, 0.61 to 0.98 on the held-out images here (other runs saw 0.24 to 0.97), much more than on seen combinations (0.82 to 0.90). on average the held-out digits did about as well as the rest in this run, but with that spread, and 90 images (a standard error of about 0.04 per model on top), the comparison isn't settled. the seed-0 model's retrievals for "a green seven" and "a red three" are green sevens and red threes.

**why:** because colour and digit vary independently in the data, the cheapest way for the encoders to align images and captions is to represent the two attributes separately: one direction (or subspace) for colour, another for digit identity, so a caption embedding is roughly "colour part + digit part". a green seven is then just the green direction plus the seven direction, both learned from other combinations. real compositionality is much harder: in web data attributes are entangled with objects and with each other, captions often don't mention the relevant attribute, and relations ("a mug on a book" versus "a book on a mug") need word order that contrastive training on bags of captions largely ignores. that's what Winoground (Thrush et al., 2022) and the "bag-of-words" analysis of Yuksekgonul et al. (2023) document. our toy world removes exactly those difficulties.
