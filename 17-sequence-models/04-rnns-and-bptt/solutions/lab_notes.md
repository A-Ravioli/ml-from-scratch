# RNNs & BPTT · lab notes

numbers are measured with the reference solution on a laptop CPU; the lab uses only synthetic data, so it gives the same results offline. it takes about 40 s, almost all of it in experiment 2.

## experiment 1: how far back does the gradient reach?

**what people usually predict:** that with $\rho = 1$ the gradient is preserved (the matrix preserves lengths), and that tanh's saturation keeps gradients bounded even for large $\rho$.

**what happens:** $\|\partial L/\partial h_t\|$ for $L = \sum h_{50}$ (64 units, mean over 32 sequences; at $t = 50$ it is always 8):

| nonlinearity | ρ | t = 1 | t = 25 | t = 40 | t = 49 |
|---|---|---|---|---|---|
| tanh | 0.5 | 2.6e-19 | 9.2e-10 | 8.2e-04 | 3.2 |
| tanh | 0.9 | 4.9e-08 | 4.7e-04 | 0.17 | 5.5 |
| tanh | 1.0 | 1.8e-06 | 3.3e-03 | 0.34 | 5.9 |
| tanh | 1.5 | 1.3 | 2.8 | 5.5 | 7.7 |
| tanh | 3.0 | 1.0e+07 | 9.6e+03 | 149 | 10.4 |
| relu | 0.5 | 3.3e-22 | 3.0e-11 | 2.1e-04 | 2.8 |
| relu | 0.9 | 5.3e-10 | 5.2e-05 | 0.07 | 4.9 |
| relu | 1.0 | 7.2e-08 | 6.9e-04 | 0.22 | 5.6 |
| relu | 1.5 | 47 | 27 | 15 | 8.3 |
| relu | 3.0 | 6.1e+16 | 1.4e+09 | 1.9e+04 | 16.8 |

at $\rho = 1$ the gradient at step 1 is $2 \times 10^{-7}$ of its size at step 50 for tanh (1.8e-06 against 8), and $9 \times 10^{-9}$ for ReLU. at $\rho = 3$ the tanh network's gradient at step 1 is **over a million times** larger than at step 50 ($1.0 \times 10^7$ against 8, a factor of $1.25 \times 10^6$), despite saturation. the curves are straight lines on the log scale: the decay or growth is exponential, at a rate fixed by $\rho$ and the nonlinearity. only near $\rho = 1.5$ does the gradient survive 50 steps roughly intact.

**why:** each step multiplies the gradient by $\operatorname{diag}(\phi'(a_s))W_{hh}^\top$. with $\rho = 1$ the matrix preserves lengths, but the diagonal shrinks every component: tanh's derivative $1 - h^2$ is below 1 for every unit that isn't exactly at 0 (it averages 0.69 here, and the measured decay, a factor $2 \times 10^{-7}$ over 49 steps, is about 0.73 per step), and ReLU's is exactly 0 for the roughly half of the units that are inactive. so a length-preserving matrix is not enough; the network would need $\rho$ somewhat above 1 to compensate, exactly as the $\rho = 1.5$ curve shows. at $\rho = 3$ almost half the tanh units are saturated at any moment (47% have derivative below 0.1), but 11% are still in the nearly linear range (derivative above 0.9), and the matrix amplifies by 3 along every direction, so the product still grows by roughly $3 \times (\text{fraction of active units})$ per step. the window of $\rho$ in which gradients neither vanish nor explode is narrow, it depends on the input statistics, and training moves $W_{hh}$ around freely, so a vanilla RNN can't be expected to stay in it. that's the motivation for gates (topic 05), which make the gradient path an explicitly controlled, near-identity map.

## experiment 2: truncation length against memory

**what people usually predict:** a smooth rise in accuracy with the truncation length, or that carrying the hidden state makes the truncation length irrelevant.

**what happens:** accuracy on the copy halves (chance 12.5%), 2 seeds:

| seq_len | accuracy |
|---|---|
| 3 | 0.27, 0.31 |
| 5 | 0.38, 0.32 |
| 6 | 0.74, 0.83 |
| 12 | 0.89, 0.80 |
| 24 | 0.96, 0.92 |

a jump at exactly the dependency length: below 6, the model stays near 30%; at 6 it jumps to around 80%, and it keeps improving with longer truncation (96% at 24).

**why:** to learn the copy, the model has to *store* each symbol in its hidden state and keep it for 6 steps, and the only signal that teaches it to do so is the gradient of the copy prediction 6 steps later, flowing back to the step where the symbol was read. with truncation length $L < 6$ that gradient path is always cut by a detach: the symbol and its copy are never in the same chunk. the carried state still contains *some* information about older symbols (a random recurrent network retains a fading echo of its inputs, the idea behind echo state networks), and the read-out layer, which is trained at every step, learns to decode a little of it. that is why accuracy at $L = 3$ is above chance (about 0.29) but far from solving the task. at $L = 6$, only the last target of each chunk has its source inside the chunk, which is already enough to learn the storage; longer chunks put more source-target pairs inside the same chunk and the copy improves further. truncated BPTT trades gradient horizon for memory and update frequency; set the truncation at least as long as the dependencies you want learned, and for a character model of Shakespeare that means at least the length of a phrase or line (the capstone uses 128–256).
