# RNNs & BPTT

## why this matters

an n-gram model forgets everything more than $n - 1$ tokens back, and making $n$ larger makes its counts useless (topic 01). word vectors fix the sparsity but not the memory: a window is still a window. what you want is a model that reads a sequence one token at a time and keeps a running summary, a fixed-size vector that can, in principle, hold anything relevant from the whole past. that is a **recurrent neural network**: the same small network applied at every step, taking the new input and its own previous state. it has a fixed number of parameters for sequences of any length, and its "context" has no hard limit.

in practice the limit is training. the gradient that teaches step 100 to remember something from step 1 has to travel back through 99 applications of the same matrix, and repeated multiplication either shrinks it to nothing or blows it up. this topic builds the RNN twice, in NumPy with backpropagation through time (BPTT) written by hand, so that you see exactly where that product of matrices comes from, and as a PyTorch module. then you'll measure the gradient fading or exploding across time steps, and train a character-level language model the way every RNN language model was trained: with *truncated* BPTT, and a hidden state carried across chunks of a long text.

## warm-up: predict first

1. an RNN with 64 hidden units reads 1,000 steps. how many sets of weights does it have: one, or 1,000?
2. multiply a vector by $0.9$ fifty times. by what factor has it shrunk? by $1.1$ fifty times?
3. the derivative of $\tanh$ is at most 1. can a tanh RNN's gradient still explode?
4. you train on chunks of 32 characters but carry the hidden state from one chunk to the next. can the model *use* information from 100 characters ago? can it *learn* to?
5. in `torch.nn.RNN`, how many bias vectors does each layer have?

## prerequisites check

- [ ] you can backpropagate through a computation graph by hand, including a tanh layer (→ ch 14 · topic 06)
- [ ] you know `nn.Module`, `nn.Linear`, `torch.autograd` and a training loop in PyTorch (→ ch 15 · topic 07)
- [ ] you know gradient clipping by global norm (→ ch 15 · topic 04)
- [ ] you know operator norms and that $\|AB\| \le \|A\|\|B\|$ (→ ch 02 · topic 08)
- [ ] you know cross-entropy for next-token prediction and perplexity (→ topic 01)

## core ideas

### idea 1: one network, applied at every step

an Elman RNN (Elman, 1990) keeps a hidden state $h_t \in \mathbb{R}^H$ and updates it with every input $x_t \in \mathbb{R}^D$:

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b),\qquad t = 1, \dots, T,$$

starting from a given $h_0$ (usually zeros). $W_{xh}$ is $H \times D$, $W_{hh}$ is $H \times H$, and the **same** weights are used at every step (warm-up 1): the network is a loop, and running it on a sequence of length $T$ is the same as building a $T$-layer feed-forward network with tied weights, the **unrolled** graph. to make predictions, a read-out layer maps each $h_t$ to outputs (logits over the next token, in a language model).

in code, with a batch of $B$ sequences, `x` is $(B, T, D)$ and the weights use `nn.Linear`'s layout (output dimension first), so the update is `h = np.tanh(x[:, t] @ W_xh.T + h @ W_hh.T + b)`. your `rnn_forward` returns all the hidden states $h_1, \dots, h_T$ as a $(B, T, H)$ array: the backward pass needs every one of them.

> **▶ build: stage 1.** implement `rnn_forward(x, h0, W_xh, W_hh, b)` in NumPy.

### idea 2: backpropagation through time

suppose the loss is a sum of per-step losses, $L = \sum_t \ell_t(h_t)$, and you're given $\delta_t = \partial\ell_t/\partial h_t$ for every step (the read-out layer's backward pass produces these; here they're the input `dhs`). $h_t$ affects $L$ in two ways: directly through $\ell_t$, and indirectly through $h_{t+1}$, which depends on it. so the total gradient flowing into $h_t$ is

$$g_t = \frac{\partial L}{\partial h_t} = \delta_t + \Big(\frac{\partial h_{t+1}}{\partial h_t}\Big)^{\!\top} g_{t+1},$$

computed backwards from $g_T = \delta_T$. with $a_t = W_{xh}x_t + W_{hh}h_{t-1} + b$ and $h_t = \tanh(a_t)$, step $t$'s backward is

$$\frac{\partial L}{\partial a_t} = g_t \odot (1 - h_t^2),\quad \frac{\partial L}{\partial W_{xh}} \mathrel{+}= \frac{\partial L}{\partial a_t}^{\!\top} x_t,\quad \frac{\partial L}{\partial W_{hh}} \mathrel{+}= \frac{\partial L}{\partial a_t}^{\!\top} h_{t-1},\quad \frac{\partial L}{\partial x_t} = \frac{\partial L}{\partial a_t}W_{xh},$$

and the gradient handed to the previous step is $\frac{\partial L}{\partial a_t}W_{hh}$ (in batch form, row vectors times matrices). the weight gradients **accumulate** over steps: the same weights were used at every step, so their gradient is the sum of the contributions of every use (on paper 1). the loop that walks $t = T, \dots, 1$ is **backpropagation through time** (Werbos, 1990), nothing more than ordinary backprop on the unrolled graph. the last gradient passed back is $\partial L/\partial h_0$.

your `rnn_backward(x, h0, hs, W_xh, W_hh, dhs)` returns a dict with `dW_xh`, `dW_hh`, `db`, `dh0` and `dx`. the tests compare it with PyTorch's autograd in float64.

> **▶ build: stage 2.** implement `rnn_backward`.

### idea 3: the same thing as a PyTorch module

`mlfs.seq.RNN(input_size, hidden_size, *, num_layers=1, nonlinearity="tanh")` is a batch-first, multi-layer RNN built from `nn.Linear` layers, written to compute exactly what `torch.nn.RNN` computes:

- layer $k$ has `ih[k]`, an `nn.Linear(in, H)` (with `in` = $D$ for the first layer and $H$ above it), and `hh[k]`, an `nn.Linear(H, H)`. both have biases, like `nn.RNN`, which keeps two bias vectors per layer (warm-up 5). two biases are redundant (only their sum matters) but matching the layout means the tests can copy `nn.RNN`'s weights into yours and compare outputs.
- the nonlinearity is `"tanh"` or `"relu"`; anything else is a `ValueError`.
- `forward(x, h0=None)` takes $(B, T, D)$ and an optional $(L, B, H)$ initial state (zeros if `None`). each layer runs over the **whole** sequence, and its outputs $(B, T, H)$ are the next layer's inputs. it returns the top layer's outputs and `h_n`, the stack of every layer's final state, $(L, B, H)$.

one cheap speed-up: the input projection doesn't depend on the recurrence, so compute `self.ih[k](layer_in)` for all $T$ steps in one call before the loop, and only do `self.hh[k](h)` inside it. the Python loop over time is still the bottleneck (it's why `nn.RNN` and `nn.LSTM`, written in C++, are much faster), but your version is fine for the sizes in this chapter, and autograd gives you BPTT for free.

> **▶ build: stage 3.** implement `RNN.__init__` and `RNN.forward`.

### idea 4: vanishing and exploding gradients

unroll the recursion of idea 2. the part of $\partial L/\partial h_t$ that comes from the loss at step $T$ is a product of Jacobians,

$$\frac{\partial h_T}{\partial h_t} = \prod_{s=t+1}^{T}\frac{\partial h_s}{\partial h_{s-1}} = \prod_{s=t+1}^{T}\operatorname{diag}\big(\phi'(a_s)\big)W_{hh}.$$

that product of $T - t$ matrices is the whole story (Bengio, Simard & Frasconi, 1994; Pascanu, Mikolov & Bengio, 2013). taking norms, $\|\partial h_T/\partial h_t\| \le (\gamma\,\|W_{hh}\|)^{T-t}$, where $\gamma$ bounds $|\phi'|$ ($\gamma = 1$ for tanh and ReLU). so:

- if $\gamma\|W_{hh}\| < 1$, the gradient from step $T$ to step $t$ **vanishes** exponentially in $T - t$ (warm-up 2: $0.9^{50} \approx 0.005$). the network still trains, but only on short-range dependencies: the signal saying "you should have remembered step 1" never arrives.
- if the largest singular value of $W_{hh}$ is above 1, the product *can* grow exponentially and the gradient **explodes**. tanh's derivative is below 1, but not for the units near zero: when $W_{hh}$ is large, some directions keep growing through the unsaturated units (warm-up 3).

the derivative factor matters as much as the matrix. tanh's $\phi'(a) = 1 - \tanh^2(a)$ is below 1 everywhere except at 0, so even a length-preserving $W_{hh}$ loses gradient at every step; ReLU's is 0 for every inactive unit. the remedies come in this chapter and the last one: **gradient clipping** (chapter 15) caps the step when the gradient explodes, without fixing vanishing; **gated** architectures (topic 05) create paths whose Jacobian is close to the identity; and attention (topic 06, chapter 18) shortcuts the path altogether.

`hidden_grad_norms(model, x)` measures it. for a one-layer `RNN`, unroll the recurrence yourself (using `model.ih[0]` and `model.hh[0]`), keep the list of hidden states, and ask autograd for the gradient of $L = \sum h_T$ (summed over the batch) with respect to *every* $h_t$ at once: `torch.autograd.grad(hs[-1].sum(), hs)`. return the per-example norm, averaged over the batch, for each step. at $t = T$ it is exactly $\sqrt{H}$, the norm of a vector of ones.

> **🔮 predict:** a tanh RNN with every singular value of $W_{hh}$ equal to 1, and random inputs. after 50 steps, is the gradient at step 1 about the same size as at step 50, or smaller by orders of magnitude?

> **▶ build: stage 4.** implement `hidden_grad_norms`.

### idea 5: truncated BPTT on a long text

a language model is trained on one long text, a million characters. backpropagating through all of it at once would need every hidden state in memory and give a single update per pass. the standard compromise is **truncated BPTT**:

1. `batchify(ids, batch_size)` cuts the text into `batch_size` **contiguous streams**, a $(B, N)$ int64 tensor whose row $b$ is the $b$-th stretch of $N = \lfloor\text{len}/B\rfloor$ characters (the tail is dropped). the rows are processed in parallel.
2. `tbptt_batches(data, seq_len)` cuts the columns into consecutive chunks: inputs `data[:, i:i+L]`, targets `data[:, i+1:i+1+L]` (the next character), for $i = 0, L, 2L, \dots$, with a shorter last chunk.
3. `train_tbptt` processes the chunks **in order**, passing each chunk's final hidden state into the next chunk, so the forward pass sees an unbroken stream, but **detaching** it first (`h = h.detach()`), so the backward pass stops at the chunk boundary.

the hidden state carries information arbitrarily far forward; the gradient flows back at most `seq_len` steps (warm-up 4). so the model can *use* long-range information if it happens to store it, but it can only *learn* to store information for about `seq_len` steps. without the detach, `backward()` would try to run through every earlier chunk (and fail, because their graphs were freed); without carrying the state, every chunk would start from zeros and the model would never see context across boundaries. each epoch starts again with `h = None`.

`CharRNN(vocab_size, hidden_size, *, num_layers=1)` is the model: `embed` (an `nn.Embedding(V, H)`, unlocked in topic 03), your `RNN` as `rnn`, and a `head` `nn.Linear(H, V)`; `forward(idx, h=None)` returns `(logits, h_n)`. `train_tbptt` uses Adam, cross-entropy over all positions, and `nn.utils.clip_grad_norm_` (the library version of chapter 15's `clip_grad_norm`, now unlocked) when `clip` is set. it returns the loss of every chunk.

> **🔮 predict:** a sequence's second half copies its first half, 6 steps later. you train with truncation length 3, carrying the state. can the model learn the copy?

> **▶ build: stage 5.** implement `batchify`, `tbptt_batches`, `CharRNN` and `train_tbptt`.

## worked example

**forward.** $D = H = 1$, $W_{xh} = 1$, $W_{hh} = 0.5$, $b = 0$, $h_0 = 0$, inputs $x = (1, 0)$:

$$h_1 = \tanh(1) = 0.76159,\qquad h_2 = \tanh(0.5 \times 0.76159) = \tanh(0.38080) = 0.36340.$$

**backward** for $L = h_2$ (so $\delta_1 = 0$, $\delta_2 = 1$):

| step | $g_t$ | $\partial L/\partial a_t = g_t(1 - h_t^2)$ | contributions |
|---|---|---|---|
| 2 | 1 | $1 \times (1 - 0.13206) = 0.86794$ | $dW_{hh} \mathrel{+}= 0.86794 \times h_1 = 0.66102$; $dx_2 = 0.86794$; $db \mathrel{+}= 0.86794$ |
| 1 | $0.86794 \times 0.5 = 0.43397$ | $0.43397 \times (1 - 0.58003) = 0.18226$ | $dW_{xh} \mathrel{+}= 0.18226 \times x_1 = 0.18226$; $dW_{hh} \mathrel{+}= 0.18226 \times h_0 = 0$; $db \mathrel{+}= 0.18226$ |

so $dW_{xh} = 0.18226$ (the $x_2 = 0$ term adds nothing), $dW_{hh} = 0.66102$, $db = 1.05020$, $dx = (0.18226, 0.86794)$, and $dh_0 = 0.18226 \times 0.5 = 0.09113$. notice how much smaller the step-1 gradient is than the step-2 one: one step back cost a factor $0.5 \times 0.42 = 0.21$.

**batching.** `batchify(range(23), 4)` keeps 20 ids as four rows of 5: `[0..4]`, `[5..9]`, `[10..14]`, `[15..19]`. `tbptt_batches(data, 2)` gives two chunks: inputs columns 0–1 and 2–3, targets columns 1–2 and 3–4. the tests check all of these.

## on paper

1. **why the weight gradient is a sum.** write the unrolled network with separate copies $W^{(1)}, \dots, W^{(T)}$ of $W_{hh}$, one per step, all equal. show that $\partial L/\partial W_{hh} = \sum_t \partial L/\partial W^{(t)}$.

<details><summary>solution</summary>

$L$ depends on $W_{hh}$ only through the copies: $L(W_{hh}) = \tilde L(W^{(1)}, \dots, W^{(T)})$ evaluated at $W^{(t)} = W_{hh}$. by the multivariable chain rule, $\frac{\partial L}{\partial W_{hh}} = \sum_t\frac{\partial\tilde L}{\partial W^{(t)}}\frac{\partial W^{(t)}}{\partial W_{hh}} = \sum_t\frac{\partial\tilde L}{\partial W^{(t)}}$. each term is the gradient a normal feed-forward layer would get, $\big(\partial L/\partial a_t\big)^\top h_{t-1}$. this is the general rule for any shared parameter, convolutions included.

</details>

2. **a bound on the gradient.** show that $\|\partial h_T/\partial h_t\|_2 \le (\gamma\sigma_\max)^{T-t}$, where $\sigma_\max$ is the largest singular value of $W_{hh}$ and $|\phi'| \le \gamma$. what does this guarantee when $\gamma\sigma_\max < 1$? does $\sigma_\max > 1$ guarantee explosion?

<details><summary>solution</summary>

each factor $\operatorname{diag}(\phi'(a_s))W_{hh}$ has norm at most $\|\operatorname{diag}(\phi')\|\,\|W_{hh}\| \le \gamma\sigma_\max$ (a diagonal matrix's norm is its largest entry), and norms are submultiplicative. so for $\gamma\sigma_\max < 1$ the gradient from step $T$ to step $t$ vanishes at least exponentially: a sufficient condition (Pascanu et al., 2013). the converse fails: $\sigma_\max > 1$ is *necessary* for explosion but not sufficient, because the diagonal factors can shrink the growing directions (saturated tanh units, inactive ReLUs). the lab's $\rho = 1.5$ tanh network shows a gradient that neither vanishes nor explodes over 50 steps.

</details>

3. **the memory of BPTT.** a 2-layer RNN with $H = 512$ processes a batch of 64 sequences of length 1,000 in float32. how much memory do the stored hidden states take for full BPTT? with truncation length 100?

<details><summary>solution</summary>

$2 \times 512 \times 64 \times 1000$ floats $= 65.5$ million $\times$ 4 bytes $\approx 262$ MB, plus as much again for intermediate values like the pre-activations. with truncation 100, ten times less, 26 MB, and ten updates per pass instead of one. memory and update frequency both favour truncation; what you lose is gradient signal for dependencies longer than the truncation length.

</details>

## experiment

`lab.py` uses your `RNN`, `hidden_grad_norms`, `CharRNN` and `train_tbptt`:

1. gradient norms across 50 time steps for recurrent matrices with every singular value equal to $\rho$, tanh and ReLU, with a slider for $\rho$.
2. a copy task whose dependency spans 6 steps, trained with truncated BPTT at lengths from 3 to 24.

## check your understanding

1. why does `rnn_backward` need `hs`, the hidden states from the forward pass?

<details><summary>answer</summary>

the tanh derivative at step $t$ is $1 - h_t^2$, and the gradient of $W_{hh}$ at step $t$ needs $h_{t-1}$. both come from the forward pass. that's why BPTT's memory grows with the sequence length: every state has to be kept until the backward pass reaches it.

</details>

2. your RNN trains, but the loss plateaus at the level of a bigram model. what does that suggest, and how would you check?

<details><summary>answer</summary>

the model is using only the last input or two: longer dependencies aren't being learned, either because gradients vanish over a few steps or because the truncation length is too short. check with `hidden_grad_norms` on the trained model (how fast does the gradient decay?) and try a longer `seq_len` or a gated cell (topic 05).

</details>

3. what goes wrong if you forget `h = h.detach()` between chunks? if you forget to carry `h` at all?

<details><summary>answer</summary>

without the detach, the second chunk's graph includes the first chunk's; `backward()` either fails ("trying to backward through the graph a second time") or, with `retain_graph=True`, costs more and more each chunk: full BPTT through the whole text. without carrying the state, every chunk starts from zeros; the model never sees context across chunk boundaries, and characters at the start of each chunk are predicted from almost nothing.

</details>

4. why does gradient clipping help with exploding gradients but not with vanishing ones?

<details><summary>answer</summary>

clipping rescales a too-large gradient to a safe length, keeping its direction, so one exploding batch can't wreck the weights. a vanishing gradient is tiny in the long-range components but not overall (the short-range terms dominate), so there is nothing to rescale, and scaling the whole gradient up wouldn't restore the missing long-range information anyway.

</details>

5. a tanh RNN has $\|W_{hh}\|_2 = 0.9$. what's the most the loss at step 100 can depend on the input at step 50?

<details><summary>answer</summary>

$\|\partial h_{100}/\partial h_{50}\| \le 0.9^{50} \approx 0.005$, times $\|W_{xh}\|$ and the tanh derivative at step 50. so a change to $x_{50}$ moves $h_{100}$ by at most half a percent of what it does to $h_{51}$: in effect the model can't learn dependencies across 50 steps, even though nothing in its architecture forbids them.

</details>

6. `nn.RNN` computes $\tanh(W_{ih}x + b_{ih} + W_{hh}h + b_{hh})$ with two biases. does the second bias add any expressive power?

<details><summary>answer</summary>

no: only $b_{ih} + b_{hh}$ matters, so the model is the same with one bias. the redundancy comes from cuDNN's conventions (and matters for GRUs, where the hidden bias sits inside the reset gate and is *not* redundant, topic 05). you keep both so weights can be copied one to one.

</details>

## common bugs

- **`x_t @ W_xh` instead of `x_t @ W_xh.T`.** symptom: shape errors when $D \ne H$, or wrong values when they're equal. `nn.Linear` stores weights as (out, in).
- **overwriting `dW` instead of accumulating it.** symptom: weight gradients that only reflect the first time step (the last one processed in the backward loop).
- **forgetting the direct term $\delta_t$.** symptom: gradients that only include the last step's loss.
- **using $1 - a_t^2$ instead of $1 - h_t^2$.** symptom: wrong gradients that look plausible; the tanh derivative is in terms of its *output*.
- **stacking layers inside the time loop wrongly.** symptom: a 2-layer model that matches `nn.RNN` for one layer but not two. the simplest correct version runs each layer over the whole sequence before the next.
- **forgetting `detach()` or resetting the state every chunk.** symptom: an error on the second chunk, or a model that can't use context across chunk boundaries.

## going deeper

- Elman (1990), "Finding structure in time", *Cognitive Science* 14(2): 179–211. [doi:10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
- Werbos (1990), "Backpropagation through time: what it does and how to do it", *Proceedings of the IEEE* 78(10): 1550–1560. [doi:10.1109/5.58337](https://doi.org/10.1109/5.58337)
- Bengio, Simard & Frasconi (1994), "Learning long-term dependencies with gradient descent is difficult", *IEEE Transactions on Neural Networks* 5(2): 157–166. [doi:10.1109/72.279181](https://doi.org/10.1109/72.279181)
- Pascanu, Mikolov & Bengio (2013), "On the difficulty of training recurrent neural networks", *ICML*. the Jacobian-product analysis and gradient clipping. [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)
- Williams & Peng (1990), "An efficient gradient-based algorithm for on-line training of recurrent network trajectories", *Neural Computation* 2(4): 490–501. truncated BPTT. [doi:10.1162/neco.1990.2.4.490](https://doi.org/10.1162/neco.1990.2.4.490)
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, chapter 10: sequence modeling, recurrent and recursive nets. [deeplearningbook.org](https://www.deeplearningbook.org)

## connections

- ← builds on: backprop by hand (ch 14 · topic 06), PyTorch modules (ch 15 · topic 07), gradient clipping (ch 15 · topic 04), operator norms (ch 02), next-token cross-entropy (topic 01), embeddings (topic 03).
- → used in: LSTMs and GRUs, which fix the vanishing gradient you measured here (topic 05); seq2seq encoders and decoders (topic 06); the capstone's stateful evaluation of a character LSTM; the long-range dependency problem that attention and transformers solve (ch 18).
