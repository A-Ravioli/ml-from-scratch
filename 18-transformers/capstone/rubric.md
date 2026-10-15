# 18 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 12 out of 15, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass | all pass, plus your own checks (e.g. cached and uncached generation agree token for token on the trained model; `token_nll` on the validation block matches `estimate_loss` within noise) |
| **results** | no comparison | test bits per character reported without an interval | test bits per character with block-bootstrap intervals for both blocks, next to chapter 17's LSTM and n-gram | as 2, plus a paired comparison (per-character losses of both models on the same test characters), and a second seed for the best model |
| **scaling** | no curve | three sizes, losses listed | the fitted exponent with a plot, and an honest statement of what three points at a fixed step budget can and can't show | as 2, plus one more size or a longer run of one size, checking a prediction the fit made |
| **analysis** | none | describes the curves | explains overfitting (or its absence) and the cache speedup on two devices, with one ablation | several ablations, a KV-cache memory estimate, and a discussion of what limits the model (context length, data, steps) |
| **clarity** | hard to follow | readable | clear figures, a results table and a narrative | a peer could rerun everything from the write-up and the config alone |
