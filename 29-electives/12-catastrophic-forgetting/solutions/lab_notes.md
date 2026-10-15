# catastrophic forgetting · lab notes

numbers measured with the reference solution on the cached real MNIST (`mlfs.datasets.mnist()`), a 784–200–200–10 ReLU MLP, Adam at $10^{-3}$, batch 64, one epoch per task, on a laptop CPU (the lab takes about a minute). uncertainties are standard deviations over seeds. `verify.py` runs the lab offline on a small synthetic stand-in for MNIST, where the numbers are different and only the code path is being tested.

## experiment 1: how fast is a task forgotten?

**what people usually predict:** gradual decay in both scenarios, with task 0 dropping below 50% somewhere in the middle or towards the end of task 1's epoch (100–190 steps), perhaps a bit faster class-incrementally.

**what happens** (seed 0 in the lab; seeds 0–4 all agree):

| scenario | before task 1 | after 10 steps | after 50 steps | end of task 1 (190 steps) | first step below 50% |
|---|---|---|---|---|---|
| class-incremental | 1.000 | 0.999–1.000 | 0.000 | 0.000 | 22–23 |
| task-incremental | 1.000 | 0.999–1.000 | 1.000 | 0.997–0.999 | never |

class-incrementally, nothing happens for about 15 steps, then task 0's accuracy falls off a cliff: below 50% at step 22 or 23 in every seed, and exactly 0% from about step 26 on. task 0 is gone after roughly 1,500 images of 2s and 3s, an eighth of an epoch. task-incrementally, the same weights train on the same data for the whole epoch and task 0 stays at 99.7% or better.

**why:** class-incrementally, every training image of a 2 or 3 pushes the logits of 0 and 1 down and those of 2 and 3 up (topic 12's on paper 1: the gradient on an absent class's logit is $p_k > 0$). at first the 0/1 logits still win on 0/1 images, so accuracy doesn't move. Adam moves every weight by about the learning rate per step, so the 2/3 logits rise at a steady rate on *every* input, including 0s and 1s, whose features overlap partly with 2s and 3s. after about 20 steps they overtake the 0/1 logits, and because the shift is nearly the same for all 0/1 images (a bias-like change), they overtake on almost all of them at once: a cliff, not a slope. after that the network never predicts 0 or 1 again; it hasn't lost the *features* for 0s and 1s, it has lost the willingness to output them. task-incrementally, the 0/1 logits are masked out of task 1's softmax and get no gradient; the hidden layers drift, but task 0's decision only compares logit 0 with logit 1, and the drift doesn't systematically favour either. experiment 2 shows that task-incremental forgetting does come, later and partially, as more tasks reshape the shared features.

## experiment 2: the whole sequence, two scenarios

**what people usually predict:** class-incremental ACC around 0.5–0.7 (the network "remembers some"), task-incremental a little higher.

**what happens** (seeds 0–2):

| scenario | ACC | BWT | FWT | forgetting |
|---|---|---|---|---|
| class-incremental | 0.196 ± 0.000 | −0.992 ± 0.000 | −0.085 ± 0.070 | 0.992 ± 0.000 |
| task-incremental | 0.917 ± 0.024 | −0.096 ± 0.030 | −0.058 ± 0.020 | 0.096 ± 0.030 |
| joint training (seed 0) | 0.960 | | | |

the class-incremental accuracy matrix is the identity, to two decimals: every task is learned (diagonal 0.97–1.00) and then lost completely. ACC is 0.196, which is $1/5$ times the last task's 0.98: at the end, the network predicts only 8s and 9s, for every input. the task-incremental network keeps 0.917 on average, 4 points under joint training. its matrix (seed 0) shows forgetting that's partial and uneven: task 0 survives task 1 untouched (1.00) and then drops to 0.58 after task 2, recovering to 0.70 by the end; task 1 ends at 0.79.

**why:** the class-incremental collapse is the cliff of experiment 1, repeated four times; the standard deviation is zero because it's not a matter of luck. the task-incremental network has two advantages at test time: it only has to choose between two classes (chance is 50%, not 10%), and the old classes never competed with the new ones during training. what it does suffer is feature drift: training on 4 vs 5 changes the hidden layers that the 0-vs-1 head reads, and nothing protects them. FWT is close to zero or negative in both scenarios because the tasks share no classes: training on 0/1 doesn't prepare a head for 2/3. class-incrementally it's negative because the untrained network predicts every class sometimes (baseline accuracy above zero), while a trained one never predicts classes it hasn't seen ($R_{j-1,j} = 0$). measured separately (seeds 0–2): masking a *class-incrementally trained* network at test time gives ACC 0.70–0.77, so telling the model the task helps a lot even without multi-head training, but training with the mask is what gets 0.92.

## experiment 3: gradual forgetting, and the learning rate

**what people usually predict:** a larger learning rate learns each task better (or the same), and either forgets more or the same.

**what happens** (ten permuted-MNIST tasks, 5,000 training images each; seed 0 in the lab, seeds 0–2 here):

| learning rate | ACC | BWT | mean new-task accuracy | task 0 at the end |
|---|---|---|---|---|
| $10^{-4}$ | 0.568 ± 0.017 | −0.097 ± 0.007 | 0.656 ± 0.017 | 0.49 ± 0.03 |
| $10^{-3}$ | 0.586 ± 0.021 | −0.298 ± 0.022 | 0.854 ± 0.001 | 0.47 ± 0.05 |
| $10^{-2}$ | 0.369 ± 0.023 | −0.536 ± 0.024 | 0.851 ± 0.005 | 0.20 ± 0.08 |

forgetting is gradual: with $10^{-3}$, task 0 falls from 0.85 to about 0.47 over nine further tasks, losing a few points per task. going from $10^{-3}$ to $10^{-2}$ changes nothing about how well each new task is learned (0.85 either way), and nearly doubles the forgetting (BWT −0.30 → −0.54). $10^{-4}$ forgets least, but only because it learns least: one epoch at that rate doesn't fit the tasks (0.66).

**why:** every permuted task uses all ten outputs, so there's no output-layer competition between old and new classes: the head for "7" is still the head for "7". forgetting comes only from the hidden layers drifting to serve new input permutations, which damages old tasks a little at a time. the learning-rate result says the weights travel much further at $10^{-2}$ than learning the new task requires: with Adam each weight moves by about the learning rate per step whatever its gradient, so ten times the rate is ten times the distance, and the extra distance is pure damage to old tasks. this is the stability–plasticity dilemma without any method: plasticity saturates (above about $10^{-3}$ more step size buys nothing), stability keeps falling. Mirzadeh et al. (2020) make the same point about learning rate, batch size and dropout as forgetting knobs.
