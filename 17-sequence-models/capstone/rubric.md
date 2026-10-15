# 17 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | fewer than half the checks pass | most checks pass | all checks pass, and the full config's LSTM beats the order-7 n-gram on the test block with a paired interval that excludes 0 | all of that, and the evaluation is clean: the checkpoint and the n-gram weights chosen on validation only, the test block used once, the state carried through the whole block |
| **results** | no test number | test bits per character without uncertainty or baseline | validation and test bits per character and perplexity for the LSTM and the n-gram, with block-bootstrap intervals and the paired difference, next to chapter 04's add-k number | all of that plus the seed spread of the LSTM (≥ 2 more seeds of the full config, or of a reduced config if compute is short, stated as such) compared with the interval's width |
| **analysis** | numbers without explanation | the learning curves described | the block-vs-i.i.d. interval comparison explained, one ablation measured against the baseline run, and the carried-vs-reset context experiment | all of that plus a mechanism for each finding (why validation turns around where it does, why the i.i.d. interval is too narrow, what long context buys and where the n-gram loses), and the validation–test gap discussed for both models |
| **clarity** | hard to follow | readable | labelled learning curves, a results table, samples at several temperatures | a reader could reproduce every number from the write-up alone (config, seeds, device, runtime) |

## what graders look for

- **one fair comparison.** the LSTM and the n-gram are scored on exactly the same characters of exactly the same blocks, in bits per character, with the LSTM's state primed from the text just before each block. a number computed on random windows, or on a different split, is not comparable with chapter 04's.
- **uncertainty that matches the data.** per-character losses are strongly correlated within a scene; an interval that treats 111,000 characters as independent is about three times too narrow (for the reference n-gram on the test block: ±0.016 bits per character against ±0.047 with blocks of 2,000). a write-up that measures how much too narrow (the `block_size=1` prompt) has understood what the interval is for.
- **validation for decisions, test once.** the best checkpoint, the n-gram's interpolation weights and any ablation are chosen on validation. the test numbers are reported once, at the end.
- **mechanisms, not adjectives.** "the LSTM is better" is a result. an explanation says *where* the gain comes from (which characters, which kinds of context) and then checks it: the carried-vs-reset experiment measures what context beyond 32 characters is worth, and comparing per-character losses of the two models on a few lines shows where the n-gram loses.
- **honest samples.** show samples at several temperatures, including a bad one, and use the copying check before claiming the model writes anything original.
