# masked modeling · lab notes

numbers are from the reference solution with the lab's defaults (real MNIST, Apple-silicon GPU via MPS). training is deterministic for a given seed on one machine; the lab took between 1.5 and 4.5 minutes depending on how busy the machine was. yours will differ in detail.

## experiment 1: how much should you hide?

**what people usually predict:** either "75%, like the paper", or "the less you hide, the easier the task, the better the features"; and that the lowest ratio reconstructs best.

**what happens:** 49 patches of 4×4 pixels, 8 epochs on 10,000 digits. kNN and linear-probe accuracy for seeds 0 / 1; reconstruction MSE on masked patches of 1,000 test digits at each model's own ratio (mean of the two seeds); the baseline fills each hidden patch with the mean of the image's visible pixels.

| mask ratio | kNN | linear probe | MAE masked MSE | baseline MSE |
|---|---|---|---|---|
| 0.25 | 0.704 / 0.725 | 0.722 / 0.741 | 0.0495 | 0.0904 |
| 0.50 | 0.730 / 0.737 | 0.750 / 0.766 | 0.0492 | 0.0909 |
| 0.75 | 0.669 / 0.721 | 0.703 / 0.733 | 0.0566 | 0.0926 |
| 0.90 | 0.610 / 0.634 | 0.624 / 0.642 | 0.0623 | 0.0973 |

an untrained MAE's encoder scores kNN 0.424 and probe 0.580, so every ratio learns something. the representation peaks at 50% here, with 25% and 75% close behind (75%'s two seeds differ by 5 points of kNN, so its rank relative to 25% isn't resolved), and 90% clearly worst. the reconstruction error is flat from 25% to 50% and rises after; every MAE beats the trivial baseline clearly, though by less as the ratio grows (45% lower error at 25% masking, 36% at 90%).

**why:** with a quarter of the patches hidden, most hidden patches have several visible neighbours, and much of each one can be continued from the strokes entering it: a local interpolation task that a model can solve without representing the digit as a whole. that's why lower ratios don't buy better features even though they're easier. at 90% only 5 patches are visible; for many digits that's not enough information to say which digit it is, the targets become mostly unpredictable, and the gradient carries little signal about shape (the reconstructions in experiment 2 turn into the average digit). in between, the task needs global reasoning but is still solvable.

He et al.'s optimum of 75% was for 224×224 ImageNet images cut into 196 patches of 16×16 pixels, with large encoders trained for hundreds of epochs and evaluated mostly by fine-tuning. a centred MNIST digit is far less redundant per patch than a natural photo (much of it is empty background, and one 4×4 patch can contain most of a stroke), and a two-block encoder trained for 8 epochs has little capacity to exploit a hard task. while preparing this lab, a run with 7×7 patches (16 per digit, a three-block encoder, 10 epochs, one seed) peaked at the *lowest* ratio, 25%: the best ratio depends on how much each patch reveals about the rest. the general lesson is the paper's, the number isn't: the mask ratio should be high enough that the task can't be solved by local interpolation.

## experiment 2: what does the MAE paint in?

**what people usually predict:** sharp digits at low masking, perhaps "hallucinated" strokes at high masking.

**what happens:** at 25% the reconstructions are close to the originals, with soft edges inside the filled patches. at 50% strokes are continued plausibly but blurrily. at 75% the overall digit is usually there, smeared; a 2 may come back as a fuzzy blob resembling an 8. at 90% almost every reconstruction is the same grey, digit-shaped smudge: the average digit, shifted slightly towards whatever the five visible patches suggest.

**why:** the loss is a per-pixel squared error, so the best prediction for an uncertain patch is the conditional *mean* of the plausible completions (on paper 2): where a stroke could go two ways, the model draws both at half intensity. the more that's hidden, the more completions are plausible, and the blurrier the mean. that makes MAE a poor image generator, and it doesn't matter: what pretraining needs is that the encoder knows which completions are plausible, and the conditional mean depends on exactly that knowledge. (generators that produce sharp samples, like the diffusion models of chapter 22, need a loss over whole distributions instead.)
