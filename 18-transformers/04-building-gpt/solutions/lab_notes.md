# building GPT · lab notes

numbers below are from the reference solution on a laptop CPU, on the real tiny Shakespeare, seed 0. the offline stand-in (random common words) gives lower losses and the same patterns.

## experiment 1: loss against model size, at a fixed number of steps

**what people predict.** the common intuition from small-data machine learning: a bigger model has more parameters to fit, so with the same few hundred steps it will be *behind*, catching up only later; or it will overfit.

**what happens.** after 300 steps (614,400 characters, less than one pass over the 892,000 training characters):

| model | parameters | train bits/char | validation bits/char |
|---|---|---|---|
| tiny ($d$ 32, 1 layer) | 16,896 | 3.62 | 3.67 |
| small ($d$ 64, 2 layers) | 108,352 | 3.38 | 3.45 |
| medium ($d$ 128, 3 layers) | 611,584 | 3.11 | 3.18 |
| large ($d$ 160, 4 layers) | 1,258,080 | **2.97** | **3.05** |

each size is better than the one below it, and the validation curves don't cross: the large model is ahead at every evaluation, from step 50 on. nothing overfits (train and validation differ by less than 0.1 bit). the gains per doubling of parameters shrink but don't stop: on a log axis the final losses fall roughly along a line. (all four are far from converged; the capstone's 7.5M-parameter GPT trains for 2,000 steps of four times as many characters (windows of 128) and reaches 2.06 bits per character on the same validation block.)

**why.** in this regime the data is effectively unlimited (no example is seen twice) and the limit is how much each step can extract. a larger model has more directions in which a gradient step can reduce the loss, and with Adam and the same learning rate each step moves every weight by about the same amount, so more parameters means a bigger effective change per step: larger models are more sample-efficient. Kaplan et al. (2020) measured exactly this ("larger models reach the same loss in fewer steps") and found the loss falls as a power law in parameters when data isn't the bottleneck. what eventually stops the trend at a fixed dataset is overfitting: once the model can memorize the 892,000 characters, more parameters help training loss and hurt validation loss, which is why compute-optimal training (Hoffmann et al., 2022, chapter 19) grows data and parameters together.

## experiment 2: how you sample matters as much as what you trained

**what people predict.** "greedy gives the best text, since it always picks the most likely character", and that the differences between the stochastic rules are cosmetic.

**what happens.** 600 characters from the prompt `ROMEO:\n`, written by the large model:

| rule | model's own bits/char | distinct 8-char chunks | copied 20-char chunks |
|---|---|---|---|
| greedy ($T = 0$) | **1.62** | **0.07** | 0.00 |
| $T = 0.8$ | 2.60 | 0.99 | 0.00 |
| $T = 1.0$ | 2.99 | 1.00 | 0.00 |
| $T = 1.5$ | 4.19 | 1.00 | 0.00 |
| top-k = 5 | 2.29 | 0.99 | 0.00 |
| top-p = 0.9 | 2.61 | 1.00 | 0.00 |

greedy decoding produces the text the model finds most probable by far (1.62 bits per character, against its 3.05 on real Shakespeare) and it is the worst text: "the the the the the stat the the stat..." with only 7% distinct chunks. $T = 1$ text has the model's own validation-level likelihood (2.99, close to 3.05, as it should: sampling at $T = 1$ produces text with the statistics the model believes in), looks like Shakespeare from a distance and is mostly invented words. $T = 1.5$ is visibly broken (4.19 bits, worse than a model that ignores context). top-k = 5 and $T = 0.8$ are the most readable; top-p = 0.9 sits close to $T = 0.8$ here. nothing is copied verbatim: a 1.3M-parameter model trained for less than one epoch can't memorize.

**why.** greedy maximizes each step's probability, and in a language model the most likely continuation of a common short word is often another common short word, whose most likely continuation is the first one again. once a cycle is the argmax of itself, a deterministic decoder can't leave it: the model sees the repetition in its context, which makes repeating even more likely (Holtzman et al., 2020 show that the probability of a repeated phrase *increases* with each repetition). the result is very probable and very bad: the most probable text isn't typical text. human text has a likelihood close to the model's average, which is exactly what $T = 1$ samples have. $T = 1.5$ moves mass into the tail, so implausible characters are picked often, and each one becomes context for the next prediction, so errors compound. top-k and top-p remove the tail (the thousands of individually unlikely, collectively likely choices) while keeping the randomness that avoids loops, and lower temperature does a similar thing more smoothly; they trade a little likelihood under the model for not derailing. the right setting depends on the use: low temperature for code or answers, higher for creative text.
