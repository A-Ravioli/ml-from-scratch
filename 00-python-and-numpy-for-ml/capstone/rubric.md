# 00 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and `correlate2d` loops only over kernel entries | all pass, plus your own extra checks (a property such as "blurring a constant image changes nothing", a pixel computed by hand, or a comparison with `np.convolve` on a single row) |
| **results** | no baseline | PSNR of the blurred images reported next to the noisy input | the denoising gain on held-out faces with its standard error, σ chosen on the tuning faces only, and the three edge F1 scores | results across the whole σ grid and several thresholds, with a clear statement of which differences exceed the uncertainty |
| **analysis** | none | describes the pictures | explains why too much blur hurts PSNR, why noise breaks the Sobel filter and blurring first fixes it, and what the padding modes do at the border | links the effects to the math (a Gaussian averages away noise with variance shrinking like $1/\sigma^2$ while smearing edges over about $\sigma$ pixels; differentiation amplifies high frequencies), or completes a stretch goal with a measured result |
| **clarity** | hard to follow | readable | clear figures with titles and labeled axes, and a narrative that answers the five prompts | a peer could reproduce every number from the write-up alone (config, seeds, σ grid, threshold, image indices, runtimes) |

## what graders look for

- **tuning on the test images.** picking the σ with the best PSNR on the same faces you report is the image-processing version of choosing $k$ on the test set. the notebook tunes on half the faces and reports on the other half; keep it that way (or use more faces).
- **honest uncertainty.** with 8 report faces, a gain of 6.9 ± 0.2 dB is clearly real; a difference of 0.2 dB between two neighbouring σ values is within the noise. say so.
- **the right comparison.** "the blurred image has a PSNR of 25 dB" means little without "the noisy input had 20 dB". always give the baseline.
- **clipping and dtypes.** sharpened images leave $[0, 1]$; displaying them without clipping (or treating uint8 photos as floats in $[0, 1]$) produces misleading pictures. mention how you handled it.
- **connecting to the chapter.** the best write-ups explain why the vectorized correlation is fast (49 whole-image operations instead of a quarter of a million small ones) and how that connects to the cost of interpreter overhead from topics 01–02.
