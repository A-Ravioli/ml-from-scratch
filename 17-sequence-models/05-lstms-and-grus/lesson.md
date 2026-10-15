# LSTMs & GRUs

## why this matters

topic 04 ended with a diagnosis: in a plain RNN, the gradient from step $T$ back to step $t$ is a product of $T - t$ Jacobians $\operatorname{diag}(\phi')W_{hh}$, and that product vanishes or explodes exponentially. you measured the gradient falling to $2 \times 10^{-7}$ of its size over 50 steps at initialization. the network isn't forbidden from remembering; it just never receives the signal that would teach it to.

the **long short-term memory** (Hochreiter & Schmidhuber, 1997) fixes this by changing what the recurrence *is*. instead of overwriting the state with a squashed function of itself at every step, an LSTM keeps a **cell state** that is updated *additively* and controlled by **gates**, small sigmoid layers that decide, per unit and per step, how much to forget, how much to write, and how much to reveal. when the forget gate is open and the input gate closed, the cell state passes through a step unchanged, and so does its gradient. the **GRU** (Cho et al., 2014) gets a similar effect with fewer parts. for twenty years these cells were what "recurrent network" meant in practice: speech recognition, translation (topic 06), handwriting, language models. in this topic you build both, check them against PyTorch to the last digit, and measure the memory each one can actually learn.

## warm-up: predict first

1. an LSTM's forget gate outputs 0.73 at every step. what fraction of the cell state survives 20 steps? 40?
2. an LSTM and a plain RNN have the same hidden size $H$ and input size $D$. how many times more parameters does the LSTM have? the GRU?
3. can an LSTM's cell state pass through 50 steps *exactly* unchanged? what would the gates have to be?
4. the GRU has no separate cell state. how can it keep information for many steps?
5. `nn.LSTM`'s forget-gate bias is initialized like every other bias, close to 0. is that a good idea?

## prerequisites check

- [ ] you built an RNN, its BPTT, and measured vanishing gradients (→ topic 04)
- [ ] you know the sigmoid and tanh and their derivatives (→ ch 14 · topic 01)
- [ ] you can write and compare PyTorch modules (→ ch 15 · topic 07)
- [ ] you know cross-entropy with ignored targets, `ignore_index=-100` (→ ch 15 · topic 07)
- [ ] you know embeddings (`nn.Embedding`) (→ topic 03)

## core ideas

### idea 1: the LSTM cell

an LSTM keeps two vectors per step, the **hidden state** $h_t$ (what the cell shows the world) and the **cell state** $c_t$ (its memory). from the input $x_t$ and the previous $h_{t-1}$ it computes four vectors of size $H$:

$$\begin{aligned}
i_t &= \sigma(W_{ii}x_t + W_{hi}h_{t-1} + b_i) &&\text{input gate: how much to write}\\
f_t &= \sigma(W_{if}x_t + W_{hf}h_{t-1} + b_f) &&\text{forget gate: how much to keep}\\
g_t &= \tanh(W_{ig}x_t + W_{hg}h_{t-1} + b_g) &&\text{candidate: what to write}\\
o_t &= \sigma(W_{io}x_t + W_{ho}h_{t-1} + b_o) &&\text{output gate: how much to show}
\end{aligned}$$

and updates

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t,\qquad h_t = o_t \odot \tanh(c_t).$$

the gates are numbers between 0 and 1 that multiply element-wise: soft switches, one per unit, recomputed at every step from the current input and state. the key line is the cell update: the old memory is *scaled* and new content is *added*. no matrix multiplies $c_{t-1}$, and no squashing nonlinearity is applied to it on its way to $c_t$. so along the cell-state path, $\partial c_t/\partial c_{t-1} = \operatorname{diag}(f_t)$, plus terms that go through $h_{t-1}$ and the gates (on paper 1). with $f_t \approx 1$ the gradient flows back through many steps almost undamaged: Hochreiter and Schmidhuber's "constant error carousel". and with $f = 1$, $i = 0$ the memory is kept exactly (warm-up 3).

**one matrix multiply per input.** all four gates use the same inputs, so implement them as two linear layers of width $4H$: `ih = nn.Linear(D, 4H)` and `hh = nn.Linear(H, 4H)`, and split `ih(x) + hh(h)` into four chunks in the order **i, f, g, o**, PyTorch's order. with that layout, `nn.LSTMCell`'s `weight_ih`, `bias_ih`, `weight_hh`, `bias_hh` copy straight into your `ih` and `hh`, which is how the tests check you.

**the forget bias.** with biases near 0, every forget gate starts at $\sigma(0) = 0.5$, and the cell state halves at every step: $0.5^{20} \approx 10^{-6}$. the network starts out as amnesiac as a plain RNN. Gers, Schmidhuber & Cummins (2000) and Jozefowicz, Zaremba & Sutskever (2015) recommend initializing the forget-gate bias to 1, so the cell starts by remembering. after building the layers, set the forget slice of `ih.bias`, positions `H:2H`, to 1 (under `torch.no_grad()`). `nn.LSTM` doesn't do this by default (warm-up 5); the lab measures how much it matters.

`LSTMCell.forward(x, state=None)` takes $x$ of shape $(B, D)$ and `state = (h, c)`, each $(B, H)$ (zeros if `None`), and returns the new `(h, c)`.

> **▶ build: stage 1.** implement `LSTMCell.__init__` and `LSTMCell.forward`.

### idea 2: the GRU cell

the **gated recurrent unit** merges the cell and hidden states and couples forgetting with writing. it computes, in PyTorch's convention,

$$\begin{aligned}
r_t &= \sigma(W_{ir}x_t + b_{ir} + W_{hr}h_{t-1} + b_{hr}) &&\text{reset gate}\\
z_t &= \sigma(W_{iz}x_t + b_{iz} + W_{hz}h_{t-1} + b_{hz}) &&\text{update gate}\\
n_t &= \tanh\big(W_{in}x_t + b_{in} + r_t \odot (W_{hn}h_{t-1} + b_{hn})\big) &&\text{candidate}\\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1}.
\end{aligned}$$

the last line is a per-unit interpolation between the old state and a candidate. with $z_t = 1$ the unit copies its state exactly; with $z_t = 0$ it replaces it. there's a single gate for "keep" and "write" ($z$ and $1 - z$), where the LSTM has two independent ones ($f$ and $i$). the reset gate decides how much of the old state the candidate may look at; when it's near 0, the unit computes its candidate from the input alone, as if starting fresh.

one detail that trips everyone up: in PyTorch's GRU (following cuDNN), the reset gate multiplies $W_{hn}h_{t-1} + b_{hn}$, **bias included**. Cho et al.'s original applies it to $h_{t-1}$ before the matrix multiply, $W_{hn}(r_t \odot h_{t-1})$. the two are different functions; you build PyTorch's, so `nn.GRUCell` weights copy into yours: `ih = nn.Linear(D, 3H)`, `hh = nn.Linear(H, 3H)`, chunks in the order **r, z, n**. compute `ih(x)` and `hh(h)` separately (not their sum), because $n$ needs $\text{hh}_n$ on its own.

> **🔮 predict:** the GRU has one gate where the LSTM has two (forget and input). on a task where the whole job is to keep one value safe from 80 steps of noise, which do you expect to learn faster?

> **▶ build: stage 2.** implement `GRUCell.__init__` and `GRUCell.forward(x, h=None)`.

### idea 3: sequence modules, and what you've unlocked

`LSTM(input_size, hidden_size, *, num_layers=1)` and `GRU(...)` stack cells exactly like topic 04's `RNN` stacked layers: a `cells` `ModuleList` (input size $D$ for the first, $H$ above), each layer running over the whole sequence and feeding its outputs to the next. `LSTM.forward(x, state=None)` takes $(B, T, D)$ and an optional `(h0, c0)`, each $(L, B, H)$, and returns `outputs` $(B, T, H)$ (the top layer's $h_t$) and `(h_n, c_n)`, each $(L, B, H)$. `GRU.forward(x, h0=None)` returns `outputs` and `h_n`. with the same weights they match `nn.LSTM` and `nn.GRU` with `batch_first=True`.

gates cost parameters: each gate is a full RNN-sized layer, so an LSTM has 4 times the parameters of an RNN of the same size and a GRU 3 times (warm-up 2; on paper 4). after this topic, `nn.LSTM`, `nn.GRU`, `nn.LSTMCell` and `nn.GRUCell` are unlocked for the rest of the course. they compute what yours compute, with fused C++ kernels that are many times faster than a Python loop over time steps. the capstone uses `nn.LSTM`.

> **▶ build: stage 3.** implement `LSTM` and `GRU`.

### idea 4: measuring memory

how long can each architecture remember, when it has to *learn* to? a clean test (in the spirit of Hochreiter & Schmidhuber's experiments) is **recall after a delay**: the first input is one of $K$ symbols, followed by `delay` distractor symbols drawn from a disjoint set; the model has to output the first symbol at the last step. nothing else is scored, so the only training signal sits `delay` steps after the information it's about, and everything in between is noise the model has to learn to ignore.

- `recall_batch(batch_size, delay, n_symbols, *, generator=None)` returns `x`, $(B, \text{delay} + 1)$ int64, with `x[:, 0]` uniform in $[0, K)$ and the rest uniform in $[K, 2K)$, and `y` of the same shape, all $-100$ (PyTorch's "ignore" target) except `y[:, -1] = x[:, 0]`. draw the distractors first, then overwrite column 0, so that everyone's batches agree for the same generator seed.
- `SeqTagger(vocab_size, hidden_size, *, cell="lstm")`: an `embed` (`nn.Embedding(V, H)`), an `rnn` (your `RNN` from topic 04, or your `GRU` or `LSTM`, chosen by `cell` in `{"rnn", "gru", "lstm"}`, else `ValueError`), and a `head` (`nn.Linear(H, V)`) producing logits at every step.
- `copy_accuracy(model, x, y)`: the fraction of positions with `y != -100` where the argmax of `model(x)` equals `y`, computed without gradients in eval mode, restoring the model's previous mode.

the training loop is a few lines (the lab and the tests write their own): cross-entropy with `ignore_index=-100` over all positions, which only counts the last.

> **🔮 predict:** with the same 400 training steps, what's the longest delay an RNN, a GRU and an LSTM (forget bias 1) will learn? does the forget bias matter?

> **▶ build: stage 4.** implement `recall_batch`, `SeqTagger` and `copy_accuracy`.

## worked example

**an LSTM step with zero weights.** set every weight and bias to 0. then every gate is $\sigma(0) = 0.5$ and the candidate is $\tanh(0) = 0$. from $c_{t-1} = 1$:

$$c_t = 0.5 \times 1 + 0.5 \times 0 = 0.5,\qquad h_t = 0.5\tanh(0.5) = 0.5 \times 0.46212 = 0.23106.$$

**perfect memory.** now set the forget-gate bias to $+20$ ($f = \sigma(20) = 1 - 2 \times 10^{-9}$) and the input-gate bias to $-20$ ($i \approx 2 \times 10^{-9}$). whatever the inputs, $c_t = c_{t-1}$ to within $10^{-7}$ per step: after 50 steps the cell state is unchanged. the tests check this.

**a GRU step with zero weights.** $r = z = 0.5$, $n = \tanh(0) = 0$, so $h_t = 0.5 \times 0 + 0.5h_{t-1}$: the state halves. with the update gate's bias at $+20$, $z = 1$ and $h_t = h_{t-1}$ exactly.

**recall batches.** `recall_batch(64, 9, 4)` gives sequences of 10 symbols: one of 0–3, then nine distractors from 4–7. `y` is `-100` in the first 9 columns and equals `x[:, 0]` in the last. if a model gets 5 of 8 scored positions right, `copy_accuracy` is $5/8$.

**parameter counts** with $D = H = 64$: an RNN layer has $64 \cdot 64 + 64 \cdot 64 + 2 \cdot 64 = 8{,}320$ parameters; a GRU layer 3 times that, 24,960; an LSTM layer 4 times, 33,280.

## on paper

1. **the cell-state gradient.** for the LSTM, compute $\partial c_t/\partial c_{t-1}$, treating $h_{t-1}$ (and hence the gates) as fixed. what is $\partial c_T/\partial c_t$ along this path? compare with the RNN's $\partial h_T/\partial h_t$.

<details><summary>solution</summary>

$c_t = f_t \odot c_{t-1} + i_t \odot g_t$ with the gates fixed gives $\partial c_t/\partial c_{t-1} = \operatorname{diag}(f_t)$, so along the cell path $\partial c_T/\partial c_t = \prod_{s=t+1}^{T}\operatorname{diag}(f_s)$: an element-wise product of gate values, with no weight matrix and no derivative of a squashing function. each unit's gradient decays like the product of its own forget gates, which the network can hold near 1. the RNN's is $\prod\operatorname{diag}(\phi'(a_s))W_{hh}$: a product of matrices whose norm is tied to $W_{hh}$'s and to $\phi' < 1$. (the full Jacobian also has paths through $h_{t-1}$ and the gates; they behave like an RNN's and can vanish, but the cell path is enough to carry the signal.)

</details>

2. **forget-bias arithmetic.** with the forget gate constant at $\sigma(b)$, what fraction of the cell state, and of its gradient, survives $d$ steps? evaluate for $b = 0, 1, 3$ and $d = 20, 40, 80$.

<details><summary>solution</summary>

$\sigma(b)^d$. $\sigma(0) = 0.5$: $10^{-6}$, $9 \times 10^{-13}$, $8 \times 10^{-25}$. $\sigma(1) = 0.731$: $1.9 \times 10^{-3}$, $3.6 \times 10^{-6}$, $1.3 \times 10^{-11}$. $\sigma(3) = 0.953$: $0.38$, $0.14$, $0.021$. at initialization, a bias of 1 leaves a usable signal over 20 steps but not 40; a bias of 3 over 40 but barely over 80. the lab finds the learnable delay tracking exactly these numbers.

</details>

3. **the GRU as a leaky integrator.** suppose $z_t = z$ is constant. unroll $h_t = (1 - z)n_t + zh_{t-1}$. how many steps does it take for the influence of $h_0$ to fall to $1/e$?

<details><summary>solution</summary>

$h_t = z^th_0 + (1 - z)\sum_{s=1}^{t}z^{t-s}n_s$: an exponential moving average of the candidates with decay $z$, the same form as chapter 15's EMA of weights. $h_0$'s weight is $z^t = e^{t\ln z}$, which reaches $1/e$ at $t = -1/\ln z \approx 1/(1 - z)$ for $z$ near 1: $z = 0.99$ remembers for about 100 steps. a unit that learns $z \approx 1$ for distractor inputs becomes a nearly perfect memory.

</details>

4. **counting parameters.** show that an LSTM layer has $4(HD + H^2 + 2H)$ parameters and a GRU layer $3(HD + H^2 + 2H)$. how many does a 2-layer LSTM with $D = 64$, $H = 512$ have?

<details><summary>solution</summary>

each gate (or candidate) has an input matrix $H \times D$, a recurrent matrix $H \times H$ and two bias vectors of size $H$ (PyTorch keeps both). the LSTM has 4 of them, the GRU 3. for the 2-layer LSTM: layer 1, $4(512 \cdot 64 + 512^2 + 1024) = 4(32{,}768 + 262{,}144 + 1{,}024) = 1{,}183{,}744$; layer 2 has $D = 512$: $4(2 \cdot 262{,}144 + 1{,}024) = 2{,}101{,}248$; 3.28 million in all. the capstone's model is about this size.

</details>

## experiment

`lab.py` trains taggers built from your cells on the recall task:

1. the longest delay an RNN, a GRU and an LSTM learn with the same training budget.
2. the LSTM's forget-gate bias at 0, 1 and 3.
3. how strongly the last hidden state depends on the first input at initialization, for each cell, as the sequence grows.

## check your understanding

1. why does the LSTM apply tanh to $c_t$ before the output gate, but not when carrying $c_t$ to the next step?

<details><summary>answer</summary>

the cell state is unbounded (it accumulates $i \odot g$ terms), so $h_t$, which feeds the gates and the next layer, is squashed to $(-1, 1)$ to keep those inputs in a sensible range. carrying $c_t$ forward *without* a nonlinearity is the whole point: a squashing function applied at every step would put a derivative below 1 into the gradient path at every step, exactly what made the RNN's gradient vanish.

</details>

2. can an LSTM's gradient explode?

<details><summary>answer</summary>

along the cell path, no: each factor is a forget gate in $(0, 1)$. but the paths through $h_{t-1}$ and the gates involve the recurrent weight matrices, like an RNN, and can grow. in practice LSTMs explode far less often than RNNs, and gradient clipping (chapter 15) handles the rest; the capstone clips anyway.

</details>

3. your LSTMCell matches `nn.LSTMCell` except for one gate. which bug produces "almost right, but the cell never forgets"?

<details><summary>answer</summary>

the chunks in the wrong order: if you read the forget gate from the slot PyTorch uses for something else (say you assumed i, g, f, o or f, i, g, o), your forget gate is computed with another gate's weights. the tests copy PyTorch's weights, so any permutation of i, f, g, o shows up as a mismatch.

</details>

4. the GRU has 25% fewer parameters than the LSTM. why might it still remember as well, or better?

<details><summary>answer</summary>

its update gate ties keeping to not-writing: $z \approx 1$ both preserves the state and blocks the candidate. an LSTM has to learn $f \approx 1$ *and* $i \approx 0$ separately; until it learns the second, the distractors keep writing into the memory. on a task like the lab's, where the whole job is to protect one stored value from noise, the coupled gate is an advantage. on others, the LSTM's separate output gate and unbounded cell help. Chung et al. (2014) and Jozefowicz et al. (2015) find neither dominates.

</details>

5. why does `copy_accuracy` switch to eval mode and back, if none of the models has dropout or batch norm?

<details><summary>answer</summary>

because it's a general evaluation function: the same code will evaluate models that do (the capstone's LSTM has dropout). evaluating in training mode would randomly drop units and understate accuracy; and not restoring the mode would silently turn dropout off for the rest of training.

</details>

6. you initialize the forget bias at 10. what could go wrong?

<details><summary>answer</summary>

$\sigma(10) \approx 0.99995$: the cell starts as a near-perfect accumulator, and every input it writes stays forever. the cell state can grow large, saturating $\tanh(c_t)$ so that $h$ stops responding, and the network has to learn to *forget*, where the gradient for lowering $f$ may be small. 1 is a compromise; 3 helps on tasks known to need long memory, as in the lab.

</details>

## common bugs

- **gate chunks in the wrong order.** symptom: a mismatch with `nn.LSTMCell`/`nn.GRUCell` in every test, though the shapes are right. PyTorch's orders: LSTM i, f, g, o; GRU r, z, n.
- **applying the GRU's reset gate to $h$ instead of to `hh(h)`.** symptom: close to `nn.GRUCell` but not equal; the bias of `hh`'s candidate chunk must be inside the product.
- **summing `ih(x) + hh(h)` before chunking in the GRU.** symptom: the candidate can't separate the two terms; mismatch with PyTorch.
- **swapping $z$ and $1 - z$.** symptom: a GRU that replaces its state when it should keep it; the zero-weights example still works (0.5 is symmetric), the comparison with PyTorch doesn't.
- **forgetting the forget bias, or setting it on both `ih` and `hh` (bias 2).** symptom: the structure test fails; in the lab, an LSTM that can't learn delays it should.
- **scoring all positions in `copy_accuracy`.** symptom: accuracies near the fraction of ignored positions; mask with `y != -100`.

## going deeper

- Hochreiter & Schmidhuber (1997), "Long short-term memory", *Neural Computation* 9(8): 1735–1780. [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
- Gers, Schmidhuber & Cummins (2000), "Learning to forget: continual prediction with LSTM", *Neural Computation* 12(10): 2451–2471. the forget gate. [doi:10.1162/089976600300015015](https://doi.org/10.1162/089976600300015015)
- Cho, van Merriënboer, Gulcehre, Bahdanau, Bougares, Schwenk & Bengio (2014), "Learning phrase representations using RNN encoder–decoder for statistical machine translation", *EMNLP*. the GRU. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
- Chung, Gulcehre, Cho & Bengio (2014), "Empirical evaluation of gated recurrent neural networks on sequence modeling". [arXiv:1412.3555](https://arxiv.org/abs/1412.3555)
- Jozefowicz, Zaremba & Sutskever (2015), "An empirical exploration of recurrent network architectures", *ICML*. ten thousand architectures, and the forget bias of 1. [proceedings.mlr.press/v37/jozefowicz15](https://proceedings.mlr.press/v37/jozefowicz15.html)
- Greff, Srivastava, Koutník, Steunebrink & Schmidhuber (2017), "LSTM: a search space odyssey", *IEEE TNNLS* 28(10): 2222–2232. which parts of the LSTM matter. [arXiv:1503.04069](https://arxiv.org/abs/1503.04069)
- Olah (2015), "Understanding LSTM networks", blog post. [colah.github.io/posts/2015-08-Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## connections

- ← builds on: the RNN, BPTT and vanishing gradients (topic 04), embeddings (topic 03), PyTorch modules and cross-entropy with ignored targets (ch 15 · topic 07), the EMA (ch 15 · topic 07).
- → used in: the GRU encoder and decoder of seq2seq with attention (topic 06); the capstone's character-level LSTM language model (with `nn.LSTM`, now unlocked); gating ideas reappear in GLU feed-forward layers and in state-space models; attention (topic 06, ch 18) removes the need to carry everything through one state.
