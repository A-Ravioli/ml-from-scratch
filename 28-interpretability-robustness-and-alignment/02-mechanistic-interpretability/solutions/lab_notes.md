# mechanistic interpretability · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU, one thread, `OMP_NUM_THREADS=1`). the data is **synthetic** by design: random tokens from a vocabulary of 48 that repeat with a random period between 4 and 16, so in-context copying is the only learnable skill (no dataset is needed, and the notebook runs the same offline). the model is your GPT with 2 layers, width 64 and 4 heads per layer, trained for 800 steps (batches of 64, AdamW, lr $10^{-3}$); the notebook takes about 20 seconds. the notes quote two more training seeds (1 and 2), run with the same recipe outside the notebook.

## experiment 1: how does in-context copying arrive?

**what people predict.** a smooth, gradually falling loss curve, with the induction score rising slowly along with it; or that the previous-token head appears first and the induction head later.

**what happens.**

| step | repeated-half loss | best induction score (layer 1) | best previous-token score (layer 0) |
|---|---|---|---|
| 0 | 3.881 | 0.07 | 0.14 |
| 100 | 3.404 | 0.08 | 0.16 |
| 140 | 3.209 | 0.10 | 0.16 |
| 180 | 2.322 | 0.42 | 0.22 |
| 220 | 1.140 | 0.71 | 0.28 |
| 260 | 0.771 | 0.76 | 0.30 |
| 300 | 0.598 | 0.76 | 0.33 |
| 400 | 0.447 | 0.75 | 0.36 |
| 800 | 0.299 | 0.77 | 0.44 |

for 140 steps the loss on the repeated half barely moves from $\log 48 = 3.87$ (the model has learned only the token frequencies), then it falls by 2.6 nats in 120 steps while the induction score jumps from 0.10 to 0.76. the previous-token score in layer 0 starts rising in the same window and keeps creeping up afterwards. seeds 1 and 2 show the same jump between steps 100 and 300 (induction score 0.08 → 0.66 and 0.08 → 0.63 between steps 100 and 200; repeated-half loss 0.25 and 0.27 at step 800).

**why.** copying needs both halves of the circuit: an induction head can't find the right key until a previous-token head has written "my predecessor was $u$" into the stream, and a previous-token head reduces the loss by almost nothing on its own (knowing the previous token doesn't predict a random next token). so for a while neither half has a gradient signal strong enough to grow; once small versions of both exist, each makes the other useful and they reinforce each other quickly. this is the phase change Olsson et al. (2022) saw in real language models, coinciding with a bump in the training loss and the onset of in-context learning. the induction score saturates near 0.77, not 1: the heads also put weight on the current token and on the first position.

## experiment 2: reading the heads

**what people predict.** one induction head in layer 1 and one previous-token head in layer 0, with the other heads doing something else.

**what happens.**

| layer | head | prev-token score | induction score |
|---|---|---|---|
| 0 | 0 | 0.39 | 0.02 |
| 0 | 1 | 0.28 | 0.01 |
| 0 | 2 | 0.36 | 0.02 |
| 0 | 3 | 0.44 | 0.01 |
| 1 | 0 | 0.07 | 0.65 |
| 1 | 1 | 0.06 | 0.67 |
| 1 | 2 | 0.06 | 0.77 |
| 1 | 3 | 0.07 | 0.76 |

every head in layer 1 is an induction head, and every head in layer 0 is partly a previous-token head. the layer-0 pattern is a two-diagonal band (the current and the previous position); the layer-1 pattern is a clean off-diagonal stripe in the second half, one position to the right of the "duplicate" diagonal $t - P$, with the first half attending mostly to position 0.

**why.** with nothing else to learn on this data, there's no pressure for heads to specialise differently, and redundancy is cheap. the layer-0 heads mix the current and previous token because the induction head needs "my predecessor" in the key at $s + 1$ but the current token too (the value it copies), and a head that writes a blend of both serves both. the induction head attends to $t - P + 1$ because K-composition matches the *query's* token against the *key's predecessor* (on paper 3).

## experiment 3: are those heads the cause?

**what people predict.** ablating the induction head destroys copying; ablating a layer-0 head does little, since it never looks at the earlier copy. in the patching grid, the matching token's position (3) lights up.

**what happens.**

| ablated | repeated-half loss |
|---|---|
| nothing | 0.304 |
| best induction head (1, 2) | 0.413 |
| all four layer-1 heads | 5.247 |
| best previous-token head (0, 3) | 1.472 |
| all four layer-0 heads | 4.609 |

removing the best induction head costs 0.11 nats; removing all four makes the model worse than guessing (5.2 > 3.87). removing a single layer-0 head costs 1.2 nats, ten times as much as the best induction head. the patching grid (metric on the clean run +6.87, corrupted −6.93): restoring the embedding or block 0's output at position 4 brings back the clean prediction completely, as does block 1's output at position 13; every other cell stays at the corrupted value, including position 3.

**why.** the four induction heads are redundant: each one alone copies the right token, so any three suffice, and single ablations under-report how much the group matters (on paper 4). the layer-0 heads are less redundant: they also carry the current token's identity forward, which every layer-1 head's query and value need, so losing one damages all four induction heads at once. all four gone, the model still "expects" to copy but has nothing to copy from, and becomes confidently wrong. the grid shows where the information travels: the corrupted token at position 4 stays at position 4 through block 0 (block 0's position-4 output fully determines the answer), and is moved to position 13 by a layer-1 head, so block 1 at position 13 suffices too. position 3, the earlier `x_3` that the query matches, is identical in both runs and so can't restore anything: patching measures where the *difference* between the inputs flows.
