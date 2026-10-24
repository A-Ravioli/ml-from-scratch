# positional information

## why this matters

the last experiment of topic 01 showed something alarming: shuffle the tokens of a sentence, feed them to self-attention, and you get exactly the same outputs, shuffled. without help, a transformer reads "the dog bit the man" and "the man bit the dog" as the same bag of words. an RNN never had this problem: it reads tokens in order, so position is built into its computation. attention threw that away to gain parallelism, and now position has to be put back in by hand.

there are many ways to do it, and the choice matters more than it looks. it decides whether a model trained on 2,048-token documents can read a 4,096-token one, whether the KV cache of topic 06 can reuse old keys, and how many parameters you spend. the original transformer *added* a fixed pattern of sines and cosines to each token's embedding. GPT-2 and BERT learned a vector per position instead. most open LLMs today (Llama, Mistral, Qwen, and chapter 27's model) *rotate* queries and keys by an angle proportional to position (RoPE). ALiBi skips vectors entirely and penalizes attention to distant tokens. this topic builds all of them, and the lab finds out which survive reading text eight times longer than anything they were trained on.

## warm-up: predict first

1. you give each position a learned vector, for positions 0 to 511. what happens at position 600?
2. a clock's hour, minute and second hands together tell you the time of day exactly. what do the sines and cosines of a sinusoidal position encoding have in common with them?
3. you rotate a query and a key in the plane, each by an angle proportional to its own position. their dot product depends on which of: both positions, only their difference, neither?
4. if *every* token of a sequence is the same token, which positional schemes can still tell position 3 from position 7?
5. ALiBi subtracts a penalty proportional to distance from every attention score. what kind of text would such a model be bad at?

## prerequisites check

- [ ] you can multiply complex numbers in polar form, and write a 2-D rotation as a matrix (→ ch 01, ch 02)
- [ ] you remember $\cos(a - b) = \cos a\cos b + \sin a \sin b$ (→ ch 01)
- [ ] your `scaled_dot_product_attention` accepts a float mask as an additive bias, and `split_heads`/`merge_heads` work (→ topic 01)
- [ ] you know what permutation equivariance means for attention (→ topic 01, on paper 3)

## core ideas

### idea 1: absolute positions, learned or sinusoidal

the simplest fix gives each position $p$ its own vector $\text{PE}(p) \in \mathbb{R}^d$ and adds it to the token embedding: $x_p = E[\text{token}_p] + \text{PE}(p)$. now "dog at position 1" and "dog at position 4" are different vectors, and attention can tell them apart. GPT-2 and BERT simply *learn* the table, one row per position up to a maximum length. that's flexible, but it costs $T_{\max} \times d$ parameters, and the table has no row for position $T_{\max}$: the model cannot even run on a longer input.

Vaswani et al. used a fixed table instead, built from sines and cosines at geometrically spaced frequencies:

$$\text{PE}(p)_{2i} = \sin(p\,\omega_i), \qquad \text{PE}(p)_{2i+1} = \cos(p\,\omega_i), \qquad \omega_i = 10000^{-2i/d}.$$

think of a clock with $d/2$ hands. hand $i$ goes around once every $2\pi/\omega_i$ positions: the first every $2\pi \approx 6.3$ positions, the last every $2\pi \cdot 10000^{(d-2)/d}$, about 47,000 positions for $d = 64$. the fast hands distinguish neighbours; the slow ones tell you roughly where in a long document you are. each hand is stored as the point $(\sin, \cos)$ on a circle, and that choice has a useful consequence: moving $k$ positions forward rotates every hand by the fixed angle $k\omega_i$, so $\text{PE}(p + k)$ is a *linear* function of $\text{PE}(p)$, the same for every $p$ (on paper 1). in particular $\text{PE}(p) \cdot \text{PE}(p + k) = \sum_i \cos(k\omega_i)$ depends only on the offset $k$: the positions know how far apart they are.

> **spec.** `sinusoidal_positions(n_positions, d_model)` returns a float32 tensor `(n_positions, d_model)` with $\sin(p\,\omega_i)$ in column $2i$ and $\cos(p\,\omega_i)$ in column $2i + 1$, $\omega_i = 10000^{-2i/d}$. an odd `d_model` ends with a sine.

> **▶ build: stage 1.** implement `sinusoidal_positions`.

### idea 2: rotary embeddings

adding positions to the token embedding has a cost: the same vector now has to carry both *what* the token is and *where* it is, mixed together in every layer. and attention only ever uses positions through the dot product $q_m \cdot k_n$, where all we usually want is the *distance* $m - n$. Su et al. (2021) asked for an encoding where that's built in: transform $q$ and $k$ so that their dot product depends on their contents and on $m - n$ only.

rotations do it. take one pair of channels and view it as a complex number $z = a + ib$. multiply the query at position $m$ by $e^{im\theta}$ and the key at position $n$ by $e^{in\theta}$, i.e. rotate each by an angle proportional to its position. the 2-D dot product of two vectors is $\operatorname{Re}(z_q \bar z_k)$, so

$$\operatorname{Re}\big(z_q e^{im\theta}\,\overline{z_k e^{in\theta}}\big) = \operatorname{Re}\big(z_q \bar z_k\, e^{i(m - n)\theta}\big),$$

which depends on positions only through $m - n$. RoPE does this for every channel pair, with the sinusoidal schedule of frequencies $\theta_i = \text{base}^{-2i/d_h}$ ($\text{base} = 10{,}000$), applied to queries and keys inside every attention layer, after the projections. there are no parameters, and nothing is added to the token's representation: the values, and the residual stream, never see a position.

which channels form a pair is a convention, and mixing conventions is a classic bug. the original paper paired neighbours $(2i, 2i + 1)$; GPT-NeoX and Llama pair channel $i$ with channel $i + d_h/2$, the **rotate-half** convention, because it's cheap to write: with $x = (x_1, x_2)$ split into halves,

$$\operatorname{RoPE}(x, m) = x \odot \cos(m\Theta) + \operatorname{rotate\_half}(x) \odot \sin(m\Theta), \qquad \operatorname{rotate\_half}(x) = (-x_2, x_1),$$

where $\Theta = (\theta_0, \dots, \theta_{d_h/2 - 1}, \theta_0, \dots, \theta_{d_h/2 - 1})$ repeats the frequencies so that channels $i$ and $i + d_h/2$ share an angle. this course uses rotate-half. the two conventions give the same model up to a permutation of channels, but weights trained with one are garbage under the other.

one more argument: the **offset**. when topic 06's cache feeds one new token at a time, that token sits at position `offset` (the number of tokens already cached), not position 0, and must be rotated accordingly. the cos/sin tables can be precomputed up to `max_seq_len`, but positions beyond it must still work: extend the table when you need to.

> **spec.** `RotaryEmbedding(head_dim, *, base=10000.0, max_seq_len=2048)` raises `ValueError` for odd `head_dim`. `forward(x, offset=0)`: `x` is `(..., T, head_dim)` (usually `(B, H, T, head_dim)`); position `offset + t` is applied to row `t`; channel $i$ and channel $i + d_h/2$ are rotated together by the angle $(\text{offset} + t)\,\theta_i$, $\theta_i = \text{base}^{-2i/d_h}$. returns a tensor of the same shape and dtype. positions at or beyond `max_seq_len` work. the cos/sin tables stay float32 even after `module.to(torch.bfloat16)` (chapter 27 runs in bf16, where a table of angles at position 3,000 would be off by whole radians) and are not part of the `state_dict`.

> **🔮 predict:** you rotate an all-ones query and an all-ones key ($d_h = 64$) to positions $m$ and $n$. at distance 0 their score is 64. what does the score look like as the distance grows? (lab, experiment 1.)

> **▶ build: stage 2.** implement `RotaryEmbedding`.

### idea 3: ALiBi, a bias on distance

Press et al. (2022) went further: why encode positions at all, if what attention needs is a preference about distance? ALiBi (attention with linear biases) adds a fixed penalty to the scores, proportional to how far back the key is:

$$s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}} - m_h\,(i - j), \qquad j \le i,$$

and $-\infty$ for $j > i$ (ALiBi was designed for causal models, so the bias includes the causal mask). each head $h$ gets its own slope $m_h$, a geometric sequence: for $H$ heads a power of two, $m_h = 2^{-8h/H}$ for $h = 1, \dots, H$, so with 8 heads the slopes are $1/2, 1/4, \dots, 1/256$. steep heads look only at the last few tokens; shallow heads see far back. for other head counts the paper takes the slopes of the nearest smaller power of two, $2^{\lfloor \log_2 H \rfloor}$, and fills the remaining heads with every other slope of the next power of two.

the bias goes into your `scaled_dot_product_attention` as a float mask, which is why topic 01 made float masks additive. no parameters, nothing added to the embeddings, and the bias at distance 600 is just a bigger number of the same kind as the bias at distance 60, so nothing the model sees at a long length is new. the price is built-in recency: a head can still attend far back, but only if the content score beats a penalty that grows without bound.

> **spec.** `alibi_slopes(n_heads)` returns the `n_heads` slopes above (a 1-D tensor, or anything `torch.as_tensor` accepts). `alibi_bias(n_heads, T, *, device=None)` returns a float32 tensor `(n_heads, T, T)` with entry $[h, i, j] = -m_h (i - j)$ for $j \le i$ and $-\infty$ for $j > i$.

> **🔮 predict:** a model trained with ALiBi on 64-token windows reads a 512-token window. will its loss at position 400 be higher, lower or the same as at position 40? and for a model with sinusoidal positions?

> **▶ build: stage 3.** implement `alibi_slopes` and `alibi_bias`.

### idea 4: one model, five schemes

to compare the schemes fairly, the lab needs one causal language model where the positional scheme is the *only* difference. you'll build a small one: token embedding, $L$ pre-norm layers each with causal multi-head attention and an MLP (a preview of topic 03's block), a final layer norm and a linear head. the scheme enters in exactly one place:

| `pos` | where position enters |
|---|---|
| `'none'` | nowhere (only the causal mask, topic 01's lab) |
| `'learned'` | a learned `(max_len, d)` table added to the embeddings; longer inputs raise `ValueError` |
| `'sinusoidal'` | `sinusoidal_positions(T, d)` added to the embeddings, for any `T` |
| `'rope'` | `RotaryEmbedding(d // n_heads)` on queries and keys in every layer |
| `'alibi'` | `alibi_bias(n_heads, T)` as the attention mask in every layer |

there's a subtle difference between the additive schemes and the other three, and the tests check it. RoPE and ALiBi change *which* tokens a query attends to, never *what* the tokens contain. if every token in the sequence is the same, every value vector is identical, and any weighted average of identical vectors is that same vector, whatever the weights. so a RoPE, ALiBi or position-free model produces the same output at every position of a constant sequence: it can't count. a learned or sinusoidal model can, because the inputs themselves differ by position. real text is never constant, so this matters less than it seems, but it tells you where positional information lives in each scheme.

> **spec.** `PositionalLM(vocab_size, *, d_model=64, n_heads=4, n_layers=2, max_len=256, pos='rope')` raises `ValueError` for an unknown `pos`. only `'learned'` has positional parameters (`pos_emb`, a `(max_len, d_model)` embedding). each layer is pre-norm: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`, with causal attention built from your topic-01 functions and a $4\times$ GELU MLP. `forward(idx)`: `(B, T)` int64 → logits `(B, T, vocab_size)`; for `'learned'`, `T > max_len` raises `ValueError`.

> **▶ build: stage 4.** implement `PositionalLM`.

## worked example

**sinusoidal**, $d = 4$, position 1: the frequencies are $\omega_0 = 1$ and $\omega_1 = 10000^{-2/4} = 0.01$, so $\text{PE}(1) = (\sin 1, \cos 1, \sin 0.01, \cos 0.01) = (0.8415, 0.5403, 0.0100, 1.0000)$.

**RoPE**, $d_h = 4$, base 100: $\theta_0 = 1$, $\theta_1 = 100^{-2/4} = 0.1$. take $x = (1, 2, 0, 0)$ at position 2, so the angles are $(2, 0.2)$. rotate-half pairs channel 0 with channel 2 and channel 1 with channel 3. pair $(x_0, x_2) = (1, 0)$ rotated by 2 rad is $(\cos 2, \sin 2) = (-0.4161, 0.9093)$; pair $(x_1, x_3) = (2, 0)$ rotated by 0.2 rad is $(2\cos 0.2, 2\sin 0.2) = (1.9601, 0.3973)$. in channel order: $(-0.4161, 1.9601, 0.9093, 0.3973)$. the formula gives the same: $x \odot \cos = (1 \cdot \cos 2, 2\cos 0.2, 0, 0)$, $\operatorname{rotate\_half}(x) = (0, 0, 1, 2)$, times $\sin = (\cdot, \cdot, \sin 2, \sin 0.2)$.

**ALiBi**, 6 heads: 6 isn't a power of two, so take the 4-head slopes $(1/4, 1/16, 1/64, 1/256)$, then every other 8-head slope, $(1/2, 1/8, 1/32, 1/128)$, until there are 6: $(1/4, 1/16, 1/64, 1/256, 1/2, 1/8)$. with 4 heads, head 1 (counting from 0) has slope $1/16$, so query 4 attending to key 1 gets bias $-3/16$.

your code must reproduce these numbers.

## on paper

1. **shifting a sinusoidal position is a rotation.** show that for each $i$ there is a $2 \times 2$ matrix $R_i(k)$, independent of $p$, with $(\text{PE}(p + k)_{2i}, \text{PE}(p + k)_{2i + 1}) = R_i(k)\,(\text{PE}(p)_{2i}, \text{PE}(p)_{2i + 1})$. conclude that $\text{PE}(p)\cdot\text{PE}(p + k)$ depends only on $k$.

<details><summary>solution</summary>

with $a = p\omega_i$, $b = k\omega_i$: $\sin(a + b) = \sin a \cos b + \cos a \sin b$ and $\cos(a + b) = \cos a\cos b - \sin a \sin b$. so

$$\begin{pmatrix}\sin(a + b)\\ \cos(a + b)\end{pmatrix} = \begin{pmatrix}\cos b & \sin b \\ -\sin b & \cos b\end{pmatrix}\begin{pmatrix}\sin a\\ \cos a\end{pmatrix},$$

a rotation by $-b$ that depends on $k$ but not on $p$. the dot product of pair $i$ is $\sin a \sin(a + b) + \cos a\cos(a + b) = \cos b = \cos(k\omega_i)$, so $\text{PE}(p)\cdot\text{PE}(p + k) = \sum_i \cos(k\omega_i)$, a function of $k$ only. (that's the constant-along-diagonals matrix in the lab.)

</details>

2. **RoPE's relative property, with matrices.** write RoPE on one channel pair as $R(m\theta)x$ with $R(\phi)$ the rotation by $\phi$. show $(R(m\theta)q)^\top (R(n\theta)k) = q^\top R((n - m)\theta)\, k$.

<details><summary>solution</summary>

$(R(m\theta)q)^\top R(n\theta)k = q^\top R(m\theta)^\top R(n\theta) k$. rotations are orthogonal, so $R(\phi)^\top = R(-\phi)$, and they compose by adding angles: $R(-m\theta)R(n\theta) = R((n - m)\theta)$. the full $d_h$-dimensional RoPE is block-diagonal with one such rotation per pair (in rotate-half order the blocks act on channels $i$ and $i + d_h/2$), so the score is a sum of terms each depending on $n - m$. since $R$ is orthogonal, RoPE also preserves norms (a test checks this).

</details>

3. **why RoPE and ALiBi can't count.** a sequence repeats one token $T$ times. show that in a model with RoPE (or ALiBi, or no positions), every layer's output is the same at every position. why does the argument fail for sinusoidal positions?

<details><summary>solution</summary>

induction over layers. at the input, every position holds the same vector $x$. suppose all positions hold the same $x$ entering a layer. the projections give the same $v$ at every position. RoPE rotates $q$ and $k$, and ALiBi changes the scores, so the attention weights $a_{ij}$ differ by position, but $\sum_j a_{ij} v = v$ because the weights of each row sum to 1. so attention outputs the same vector everywhere, and the residual connection, layer norm and MLP act per position: the layer's output is again the same at every position. with sinusoidal (or learned) positions the input is $x + \text{PE}(p)$, which differs by position, so the base case fails.

</details>

4. **RoPE's slowest hand.** with base $10{,}000$ and $d_h = 64$, what is the longest wavelength (in positions) among RoPE's frequencies, and the shortest? a model trained on sequences of 64 tokens: for which frequencies has it seen a full turn of the angle $m\theta_i$?

<details><summary>solution</summary>

$\theta_i = 10000^{-2i/64}$ for $i = 0, \dots, 31$. the wavelength is $2\pi/\theta_i$: shortest $2\pi \approx 6.3$ positions ($i = 0$), longest $2\pi \cdot 10000^{62/64} \approx 2\pi \cdot 7{,}500 \approx 47{,}000$ positions. a full turn within 64 positions needs $2\pi/\theta_i \le 64$, i.e. $10000^{2i/64} \le 10.2$, $i \le 8$: only the 9 fastest pairs. for the other 23, training only ever showed angles in a fraction of the circle, and positions 300 or 500 rotate them into angles the network has never seen, which is the usual explanation for RoPE's failure to extrapolate, and the motivation for position interpolation (Chen et al., 2023), which rescales positions so long inputs stay inside the trained range.

</details>

## experiment

the lab first looks at what positions look like to attention: the matrix of sinusoidal dot products, and the RoPE score of two identical vectors as their distance grows to 2,000. then the main event: five copies of your `PositionalLM`, identical except for the scheme, trained on 64-character windows of Shakespeare and asked to read 512-character windows. which schemes keep working past position 64, and which do worse than having no positional information at all?

## check your understanding

1. why does the sinusoidal encoding use both a sine and a cosine at each frequency, instead of just sines at $d$ different frequencies?

<details><summary>answer</summary>

with the pair $(\sin, \cos)$, shifting the position is a rotation of each pair (on paper 1), a linear map that's the same at every position, which a linear layer can learn to use. with sines alone, $\sin(a + b)$ isn't a function of $\sin a$ alone (you need $\cos a$ too), so "the position $k$ steps back" isn't a fixed linear function of the encoding.

</details>

2. learned position embeddings worked fine for GPT-2. what concretely goes wrong at position 1,024 for a model trained with `max_len = 1024`?

<details><summary>answer</summary>

there is no row 1,024 in the table: indexing fails, so the model can't be run at all. if you extended the table with random or zero vectors, those positions would get inputs the network has never seen. there's no structure linking row 1,024 to row 1,023, so nothing transfers.

</details>

3. RoPE is applied to queries and keys but not values. what would rotating the values do?

<details><summary>answer</summary>

it would put absolute position into the content each token hands over: the attention output would depend on the absolute positions of the attended tokens, not just on relative ones, breaking the property that motivated RoPE. it would also break your constant-sequence test: identical tokens would contribute different vectors.

</details>

4. with a KV cache, token 100 is fed alone with `offset=100`. what goes wrong if you forget the offset?

<details><summary>answer</summary>

the new query and key are rotated as position 0. the cached keys were rotated with their true positions 0..99, so the new token's scores against them correspond to distances $0 - n$ instead of $100 - n$: negative distances, i.e. the model believes it's at the start of the sequence, looking at the future. outputs differ from the uncached model and quality collapses. the cache tests in topic 06 catch exactly this.

</details>

5. why does ALiBi give each head a different slope instead of one shared slope?

<details><summary>answer</summary>

a single slope fixes one effective context length for the whole layer. with a geometric range of slopes, steep heads specialize in the last few tokens (local syntax), shallow heads in long-range context, all at once. Press et al. found that the geometric sequence worked robustly without tuning.

</details>

6. a model with no positional encoding at all is trained as a causal LM. how can it possibly know word order?

<details><summary>answer</summary>

the causal mask itself: position $t$ averages over exactly $t + 1$ tokens, so the first position copies itself, the second averages two, and so on. the model can learn to recover position and order from these differing averages (Haviv et al., 2022). a bidirectional model has no such asymmetry.

</details>

## common bugs

- **pairing the wrong channels in RoPE.** interleaved pairs $(2i, 2i + 1)$ instead of rotate-half $(i, i + d_h/2)$. relative-position tests still pass (it's a valid RoPE), but the explicit-rotation test fails, and so does loading weights trained with the other convention.
- **repeating the frequencies wrongly.** `torch.repeat_interleave(freqs, 2)` goes with the interleaved convention; `torch.cat([freqs, freqs])` with rotate-half.
- **forgetting the offset.** the model works on whole sequences and breaks only with a cache.
- **a hard limit at `max_seq_len`.** slicing `cos[offset:offset + T]` past the end of the table silently returns a shorter tensor, and the broadcast error appears far from the cause.
- **ALiBi with a bool mask.** passing `bias == 0` or similar turns the distances into a 0/1 pattern. the bias must stay a float tensor.
- **adding sinusoidal positions in float64 to float32 embeddings.** the result silently becomes float64, everything downstream is slower, and MPS refuses float64 outright.

## going deeper

- Vaswani et al. (2017), "Attention is all you need", *NeurIPS*, section 3.5 (sinusoidal encodings). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Shaw, Uszkoreit & Vaswani (2018), "Self-attention with relative position representations", *NAACL*. the first relative scheme. [arXiv:1803.02155](https://arxiv.org/abs/1803.02155)
- Su, Lu, Pan, Murtadha, Wen & Liu (2021), "RoFormer: enhanced transformer with rotary position embedding". RoPE, including the long-term decay of experiment 1 (section 3.4.3). [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Press, Smith & Lewis (2022), "Train short, test long: attention with linear biases enables input length extrapolation", *ICLR*. ALiBi and the extrapolation experiment the lab reproduces at small scale. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Kazemnejad, Padhi, Natesan Ramamurthy, Das & Reddy (2023), "The impact of positional encoding on length generalization in transformers", *NeurIPS*. no positional encoding extrapolates surprisingly well. [arXiv:2305.19466](https://arxiv.org/abs/2305.19466)
- Chen, Wong, Chen & Tian (2023), "Extending context window of large language models via positional interpolation". [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)

## connections

- ← builds on: attention and its permutation equivariance, float masks (topic 01); rotations and complex numbers (ch 01–02); trigonometric identities (ch 01).
- → used in: GPT's learned and sinusoidal positions (topic 04), BERT and ViT's learned positions (05, 07), RoPE offsets with the KV cache (06), Llama's RoPE and long-context extensions (ch 27), positional encodings for graphs and sets (ch 20).
