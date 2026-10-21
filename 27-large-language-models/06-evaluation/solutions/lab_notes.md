# evaluation · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU, 4 threads), tiny Shakespeare characters from the cache, a `Llama` of 4 layers and width 128 trained for 500 steps (seed 0 in the notebook). the benchmark is **synthetic**: 200 cloze items from held-out lines, chance 25%. intervals are 95% bootstraps over items; they don't include the variation between training runs, which the notes give separately (seeds 1 and 2, run with the same recipe in a separate script). offline the notebook trains for 150 steps on synthetic text and scores 60 items; its numbers mean little.

## experiment 1: one model, one benchmark, five scores

**what people predict.** that the formats agree within a few points, since the model's knowledge is the same.

**what happens.**

| format | accuracy | 95% interval | seeds 1, 2 |
|---|---|---|---|
| raw | **0.345** | [0.275, 0.410] | 0.295, 0.295 |
| raw, unnormalized | 0.295 | [0.235, 0.360] | 0.280, 0.270 |
| instruction | **0.220** | [0.165, 0.280] | 0.255, 0.255 |
| after a speaker name | 0.320 | [0.255, 0.390] | 0.305, 0.290 |
| lettered (A–D) | 0.255 | [0.195, 0.315] | 0.250, 0.250 |

the highest and lowest formats are 12.5 points apart for the same model and items; the paired difference raw − instruction is +0.125 [+0.080, +0.175], well outside its interval. the lettered format sits at chance for all three models. normalization adds 1.5–5 points. across seeds the raw score moves by 5 points: the bootstrap intervals over items (±7 points) understate the total uncertainty, because they hold the model fixed.

**why.** the score measures the model *and* the format. the instruction prompt is text this model has never seen ("Complete the line.\nLine: ... Completion:"): the continuation must follow "Completion:", a context unlike anything in Shakespeare, so the model's predictions there are poor for every ending, and the differences between endings shrink towards noise (it even dips below chance in one run, within the interval). putting a speaker name first makes the context look like the training data. the lettered format asks the model to map an ending to a letter, a skill it never learned: "A", "B", "C", "D" after "Answer:" are equally meaningless to it, so it scores chance however much it knows (Gu et al., 2024, show the same split for small open models on MMLU-style tasks, which is why their standard scores small models in the cloze form). unnormalized scoring prefers the ending with fewer characters, which carries no information about correctness. an honest report names the format and the normalization, gives the interval and the number of items, and, when formats disagree this much, reports several.

## experiment 2: contamination

**what people predict.** a modest rise, and that the leak would show up as better general quality.

**what happens.**

| | clean model | contaminated model |
|---|---|---|
| accuracy on the benchmark | 0.345 | **0.515** |
| accuracy on 177 fresh items | 0.277 | 0.299 |
| loss on held-out text (nats/char) | 1.726 | 1.716 |
| items flagged by 8-gram overlap with its training text | 0 of 200 | 200 of 200 |

the leaked copies are 8.3% of the contaminated training text, and each benchmark line passed through training about 19 times (ten copies, 1.87 epochs). the benchmark score jumps by 17 points (paired: +0.170 [+0.115, +0.230]; seeds 1 and 2: 0.295 → 0.490, +19.5 points both), while accuracy on fresh items and the held-out loss stay within noise. the 8-gram check flags every leaked item and none in the clean model's training text.

**why.** seeing the exact line about 19 times teaches the model that line's ending, which is exactly what the benchmark asks, without teaching anything about other lines: the gain is specific to the test items, so general measures (held-out loss, fresh items) can't reveal it. it doesn't reach 100% because a 0.7M-parameter model in 500 steps can't memorize 200 lines verbatim alongside everything else; a large model would. exact n-gram overlap works here because the leak is verbatim and the training text is available; it would miss a paraphrased or reformatted copy and depends on access to the training data, which is why evaluators also use fresh post-cutoff items, private test sets, and verbatim-completion probes.

## experiment 3: the model as a judge

**what people predict.** that the model, asked which line is better, does a bit better than chance; few predict how strongly it prefers one position.

**what happens.** 120 pairs of a real held-out line and its shuffled version:

| judge | right, real first | right, real second | right overall | mean P(first is better) | right after `pairwise_judge` |
|---|---|---|---|---|---|
| asking | 0.94 | 0.12 | 0.53 | 0.51 | **0.67** |
| likelihood | 0.99 | 0.99 | 0.99 | 0.50 | **0.99** |

asked directly, the model picks the real line 94% of the time when it's shown first and 12% when it's shown second: it answers "1" almost regardless of content (overall 53%). its probabilities stay close to 0.5 (mean 0.51), slightly tilted towards " 1". averaged over both orders by `pairwise_judge`, the tilt cancels and a weak genuine signal remains: 67% right. comparing the two lines' per-character log-likelihoods instead is right 99% of the time, in either order.

**why.** the "asking" judge's output is mostly a property of the prompt (which digit tends to follow "Better:") rather than of the lines, so position dominates: that's the position bias Zheng et al. (2023) and Wang et al. (2023) document for large models, here in extreme form. swapping the order and averaging removes any bias that's the same in both orders, which exposes the small part of the judgement that does depend on the lines. the likelihood judge works because "which of these is real Shakespeare?" is exactly what a Shakespeare language model measures, and it never sees a question to be biased by. it can't judge a large model's answers to open questions: likelihood under some model rewards typical text, not correct or helpful text, which is why practical LLM judges are prompted strong models, swapped and averaged, and themselves validated against human labels. swapping doesn't cancel biases that move with the content (length, style, self-preference).
