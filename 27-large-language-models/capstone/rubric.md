# 27 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 12 out of 15, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass | all pass, plus your own checks: the reference stays frozen through DPO, test lines and words never appear in SFT or preference data, and the corpus statistics add up (documents = filtered + duplicates + kept) |
| **results** | no numbers | numbers without intervals or without the base model | every stage (base, SFT, DPO) evaluated on the same held-out tasks, with bootstrap intervals, the previous stage as a win-rate baseline, and held-out loss | as 2, plus at least three seeds of post-training with the spread reported next to the intervals, and a paired comparison where you claim one stage beats another |
| **analysis** | none | describes the numbers | says which effects exceed the seed spread and which don't, separates format, skill and preference effects, and checks the DPO model's answers for gaming of the gold rule | as 2, plus one ablation you chose (no dedup, LoRA instead of full SFT, a different $\beta$, more DPO steps) with a prediction written before running it |
| **the model card** | missing | generic template | every section filled with *your* numbers, data sources (synthetic parts named as such), procedure and limitations | as 2, plus a limitation you found yourself in the model's outputs (with an example) and a statement of what the evaluation does not cover |
| **clarity** | hard to follow | readable | tables, the training curves and sample answers side by side, a narrative from data to card | a peer could rerun every number from the notebook and the card alone, and would know which differences are within noise |
