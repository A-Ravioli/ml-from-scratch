# neural audio codecs

## why this matters

a phone call in the 1970s sent 8,000 samples a second at 8 bits each: 64 kbps of μ-law PCM. modern speech codecs need a tenth of that, and **neural codecs** such as SoundStream (Zeghidour et al., 2021) and EnCodec (Défossez et al., 2022) go down to 1.5–6 kbps, a fraction of a bit per sample, by learning an encoder that squeezes the waveform into a short sequence of vectors, quantizing those vectors against learned codebooks, and learning a decoder that turns the codes back into sound.

the quantizer at their heart is **residual vector quantization** (RVQ): a cascade of small codebooks, each encoding what the previous ones missed. it solves a problem you met with the VQ-VAE in ch 21: to spend $b$ bits on a vector with a single codebook you need $2^b$ codewords, so 80 bits per frame would need $2^{80}$. RVQ spends them as 8 codes of 10 bits each, from 8 codebooks of 1,024 vectors. it also gives you a bitrate dial for free: drop the last stages and you have a coarser, cheaper code. and the codes are discrete **tokens**, which is why audio language models (AudioLM, VALL-E) generate speech and music by predicting RVQ codes the way GPT predicts text.

this topic builds RVQ in NumPy, trains its codebooks with k-means, works out what bits cost, builds a trainable RVQ from your ch 21 `VectorQuantizer`, and wraps it in a small convolutional codec. the lab ends with the block's **project**: bitrate against quality for your codec, next to classic baselines. **all audio is synthetic** (topic 24's vowels, chirps and shaped noise), and every metric you'll compute is a stand-in for what matters, human listening; the lesson is explicit about what those stand-ins miss.

## warm-up: predict first

1. μ-law telephone audio is 64 kbps. a neural codec at 6 kbps on the same 8 kHz audio spends how many bits per sample?
2. one codebook with $2^{16}$ codewords, or two RVQ stages of 256 codewords each: both cost 16 bits per vector. which gives the lower error, and which can you actually train?
3. each extra RVQ stage costs the same number of bits. does each one remove the same *amount* of error, the same *fraction*, or neither?
4. if every stage's codes looked uniformly distributed, could a lossless entropy coder make the bitstream smaller?
5. a codec reproduces the magnitude spectrum of every frame exactly but gets the phases wrong. what's its SNR? its log-spectral distance? does it sound bad?

## prerequisites check

- [ ] you've built a `VectorQuantizer` with the straight-through estimator, `codebook_usage` and `restart_dead_codes` (→ ch 21 · topic 04)
- [ ] you know Lloyd's algorithm and k-means++ initialization (→ ch 11 · topic 01)
- [ ] entropy in bits, and why it lower-bounds the average code length (→ ch 05 · topic 05)
- [ ] strided `Conv1d` and `ConvTranspose1d` (→ ch 16)
- [ ] you've built `stft` and know what a log spectrum shows (→ topic 24)

## core ideas

### idea 1: residual vector quantization

a **vector quantizer** maps a vector $x \in \mathbb R^D$ to the nearest of $K$ codewords $c_1, \dots, c_K$ and transmits its index: $\log_2 K$ bits. the error shrinks as $K$ grows, but the codebook is a list of $K$ vectors, so $b$ bits per vector need $2^b$ of them: memory, search and training all grow exponentially in the bitrate.

**residual VQ** (multi-stage VQ; Juang & Gray, 1982) uses $S$ small codebooks in sequence. stage 1 quantizes $x$; stage 2 quantizes what stage 1 got wrong; and so on:

$$r_0 = x,\qquad k_s = \arg\min_k\|r_{s-1} - c^{(s)}_k\|^2,\qquad r_s = r_{s-1} - c^{(s)}_{k_s},\qquad \hat x = \sum_{s=1}^S c^{(s)}_{k_s}.$$

the code is the tuple $(k_1, \dots, k_S)$: $S\log_2 K$ bits from only $SK$ codewords. the price is that the reachable points $\hat x$ are sums, one codeword per stage, a structured subset of what a single $K^S$ codebook could place freely, and that greedy encoding (each stage choosing its nearest codeword without looking ahead) needn't find the best tuple even among those. in exchange, stages are naturally ordered from coarse to fine, so decoding only the first $S' < S$ codes gives a valid, coarser reconstruction at $S'/S$ of the bitrate.

> **spec.** `nearest_codes(X, codebook)`: for `X` of shape `(N, D)` and `codebook` `(K, D)`, the `(N,)` int64 index of the nearest codeword by squared Euclidean distance (vectorized). `rvq_encode(X, codebooks)`: `codebooks` is a list of `(K_s, D)` arrays; returns the `(N, S)` int64 codes of the greedy cascade above. `rvq_decode(codes, codebooks)` returns the `(N, D)` sum $\sum_s c^{(s)}_{k_s}$ over the stages present in `codes` (which may have fewer columns than there are codebooks: the first $S'$ are used).

> **▶ build: stage 1.** implement `nearest_codes`, `rvq_encode` and `rvq_decode`.

### idea 2: codebooks by k-means, stage by stage

where should the codewords go? for one stage, minimizing the mean squared error $\mathbb E\|r_{s-1} - c_{k(r)}\|^2$ over the codewords is exactly the k-means objective, and Lloyd's algorithm (assign each vector to its nearest codeword, move each codeword to the mean of its vectors, repeat) finds a local optimum. RVQ is trained greedily the same way: run k-means on the data for stage 1, compute the residuals, run k-means on the residuals for stage 2, and so on. that's how EnCodec initializes its codebooks before gradient training, and it's a complete (if non-neural) codec on its own.

each stage removes part of the remaining error. residuals are smaller and more "noise-like" than the data, with less structure for a codebook of fixed size to exploit, so on real data the error usually falls by a slowly shrinking fraction per stage. a useful reference is scalar quantization: a uniform quantizer gains 6.02 dB of SNR per extra bit (on paper 2), i.e. each bit cuts the error by a factor of 4. you'll measure how RVQ compares.

> **spec.** `fit_rvq(X, n_codes, n_stages, *, n_iters=20, rng=None)`: `rng = np.random.default_rng(rng)`; for each stage, initialize with your ch 11 `kmeans_plus_plus(residuals, n_codes, rng=rng)`, run `n_iters` Lloyd iterations (a codeword with no vectors keeps its position), then replace the residuals by what that stage leaves over. returns the list of `n_stages` `(n_codes, D)` codebooks. `rvq_distortion(X, codebooks)` returns the `(S,)` array of mean squared error *per coordinate*, `mean((X - X_hat)^2)`, after $1, 2, \dots, S$ stages of greedy encoding.

> **🔮 predict:** with 16 codes per stage on 8-dimensional data, what fraction of the error does each stage remove: about the same each time, or less and less?

> **▶ build: stage 2.** implement `fit_rvq` and `rvq_distortion`.

### idea 3: what bits buy: rates, baselines and metrics

**bitrate.** a codec with hop length $H$ (samples per code frame) at sample rate $f_s$ emits $f_s/H$ frames per second, each with $S$ codes of $\log_2 K$ bits:

$$\text{bitrate} = \frac{f_s}{H}\cdot S\cdot\log_2 K\ \text{bits/s}.$$

EnCodec at 24 kHz uses $H = 320$ (75 frames/s) and $K = 1024$: 750 bits/s per stage, so 2, 4, 8, 16 or 32 stages give its advertised 1.5–24 kbps. this is the rate for sending raw indices. if some codes are more frequent than others, a lossless entropy coder can do better: the empirical **entropy** $H = -\sum_k p_k\log_2 p_k \le \log_2 K$ of the codes is what a zeroth-order entropy coder needs per code, and a model that predicts each code from its context (an **audio language model**) can need less still. EnCodec ships a small transformer language model over its codes for exactly this.

**the baseline.** classic telephony quantizes each *sample* with $b$ bits: $8000b$ bits/s at 8 kHz. plain **PCM** spaces the $2^b$ levels uniformly over $[-1, 1]$ (a *midrise* quantizer: levels at the centres of $2^b$ equal cells, so 0 isn't a level). **μ-law** (the smooth companding law that ITU-T G.711 approximates with an 8-segment piecewise-linear curve) first compresses the amplitude, $y = \operatorname{sign}(x)\ln(1 + \mu|x|)/\ln(1 + \mu)$ with $\mu = 255$, quantizes $y$ uniformly and expands back, which spends more levels on quiet sounds, where the ear is more sensitive and speech spends most of its time. at 8 bits that's essentially the 64 kbps telephone codec. the codec in this topic has to beat these at the same bitrate to be worth anything.

**metrics.** the **signal-to-noise ratio** $\text{SNR} = 10\log_{10}\big(\sum x^2/\sum(x - \hat x)^2\big)$ dB compares waveforms sample by sample. it punishes any phase error: delay a tone by half a period and the "noise" is twice the signal. the **log-spectral distance** compares log power spectra frame by frame,

$$\text{LSD} = \frac1M\sum_m\sqrt{\frac1F\sum_k\Big(10\log_{10}(|X_{k,m}|^2 + \epsilon) - 10\log_{10}(|\hat X_{k,m}|^2 + \epsilon)\Big)^2}\ \text{dB},$$

and is blind to phase within a frame, but weighs a 20 dB error in an inaudible quiet bin as heavily as one in a loud bin, unless $\epsilon$ floors the quiet bins. neither is what people hear. perceptual metrics, PESQ (Rix et al., 2001) and ViSQOL (Hines et al., 2015), model the ear and are calibrated against listening tests. they aren't available here, and the neural codec papers rely on MUSHRA listening tests anyway, because even perceptual metrics misjudge the artifacts of generative decoders. keep that in mind when you read the project's numbers: SNR and LSD can rank codecs in opposite orders, and neither ranking is the one a listener would give.

> **spec.** `bitrate(sample_rate, hop_length, n_stages, n_codes)` returns the formula above as a float (bits/s). `entropy_bits(codes, n_codes)` returns the entropy in bits of the empirical distribution of the integer array `codes` (any shape). `pcm_quantize(x, bits, *, mu=None)` clips `x` to $[-1, 1]$, optionally μ-law compresses it, quantizes to the midrise levels $(i + \tfrac12)\cdot\tfrac{2}{2^b} - 1$, $i = \lfloor (y + 1)2^{b-1}\rfloor$ clipped to $[0, 2^b - 1]$, and expands back if `mu` is given (inverse $\operatorname{sign}(q)\big((1 + \mu)^{|q|} - 1\big)/\mu$). `snr_db(ref, est)` is the SNR above. `log_spectral_distance(ref, est, *, n_fft=256, hop_length=64, eps=1e-10)` is the LSD above, using your topic 24 `stft` (here `audio.stft`).

> **▶ build: stage 3.** implement `bitrate`, `entropy_bits`, `pcm_quantize`, `snr_db` and `log_spectral_distance`.

### idea 4: a residual quantizer you can train

inside a neural codec, RVQ quantizes the encoder's output $z_e$, a `(B, D, T)` sequence of $T$ vectors, and gradients must flow through it. each stage is a VQ-VAE quantizer (ch 21) applied to the residual left by the previous stages: stage $s$ returns its code vectors $q_s$, its codebook loss $\|\operatorname{sg}(r_{s-1}) - q_s\|^2$ (pulls codewords toward the residuals they serve) and its commitment loss $\beta\|r_{s-1} - \operatorname{sg}(q_s)\|^2$ (keeps the encoder near the codewords), and the next stage sees $r_s = r_{s-1} - q_s$. the decoder gets the total $q = \sum_s q_s$ through *one* straight-through estimator,

$$z_q = z_e + \operatorname{sg}(q - z_e),$$

so the decoder's gradient reaches the encoder unchanged. (summing the stages' own straight-through outputs would hand the encoder $S$ copies of it; on paper 3.) three practical tricks from SoundStream and EnCodec make this train well. **k-means initialization**: run your `fit_rvq` on a batch of encoder outputs before training, so no codeword starts far from the data. **dead-code restarts**: a codeword nobody uses gets no codebook gradient and stays dead, so periodically move unused codewords onto random encoder outputs (your ch 21 `restart_dead_codes`). **quantizer dropout**: for each training batch, use only the first $n_q \sim \mathcal U\{1, \dots, S\}$ stages. the decoder then learns to work with any prefix of the code, and one model serves every bitrate.

> **spec.** `ResidualVQ(n_stages, n_codes, code_dim, *, beta=0.25)` is an `nn.Module` with `self.stages`, an `nn.ModuleList` of `n_stages` `mlfs.generative.VectorQuantizer(n_codes, code_dim, beta=beta)`. `forward(z_e, n_stages=None)` takes `(B, D, T)`, runs the first `n_stages` stages (all if None) on successive residuals, each stage's input being the previous residual minus that stage's (detached) code vectors, and returns `(z_e + sg(q - z_e), sum of the stages' vq losses, codes (B, n_used, T) int64)`. `decode_indices(codes)` returns the `(B, D, T)` sum of the code vectors of the first `codes.shape[1]` stages. `init_from_data(z_e, *, n_iters=10, rng=None)` reshapes `z_e` to its `(B*T, D)` vectors, sets each stage's codebook to the corresponding `fit_rvq(vectors, n_codes, n_stages, n_iters=n_iters, rng=rng)` codebook (without tracking gradients) and returns `self`.

> **▶ build: stage 4.** implement `ResidualVQ`.

### idea 5: the codec, and codec language models

the **encoder** is a stack of 1-D convolutions whose strides multiply to the hop length $H$: with strides (2, 4) every code frame summarizes 8 samples. channels grow as time shrinks, and a last convolution projects to `code_dim`. the **decoder** mirrors it with transposed convolutions. SoundStream and EnCodec add residual blocks, LSTMs and, crucially, *adversarial* losses (a discriminator judging realism), which let the decoder synthesize plausible detail instead of the blurry average. this topic leaves out the adversarial part: training it well takes hours of GPU time, and the result can't be judged without listening anyway.

the training loss here has three parts: an L1 loss on the waveform; a **multi-scale spectral loss** that compares STFT magnitudes at several window lengths, both linearly and in log, so errors are judged at several time–frequency tradeoffs (topic 24's lab); and the RVQ's codebook and commitment losses. the spectral loss is what lets a codec ignore phase details that don't matter to the ear, at least partly; the waveform term keeps it anchored.

once trained, the codec turns audio into a grid of tokens: $S$ codes per frame, $f_s/H$ frames per second. **codec language models** model that grid. AudioLM (Borsos et al., 2022) generates the coarse first stages' codes with a transformer (after a layer of "semantic" tokens from a self-supervised speech model), then the finer stages conditioned on them; VALL-E (Wang et al., 2023) does text-to-speech by predicting EnCodec codes from phonemes and a 3-second voice prompt. everything you know about GPT (ch 18) applies, with two twists: there are $S$ tokens per time step, and the coarse-to-fine order of RVQ lets models predict the important tokens first. the lab measures how predictable your codec's tokens are, which is the room an entropy coder or a language model has to work with.

> **spec.** `ConvCodec(*, channels=32, code_dim=16, strides=(2, 4, 4), n_stages=8, n_codes=256)` (strides even) has attributes `encoder` (a module mapping `(B, 1, L)` to `(B, code_dim, L / hop_length)`), `quantizer` (a `ResidualVQ(n_stages, n_codes, code_dim)`), `decoder` (mapping back to `(B, 1, L)`) and `hop_length = prod(strides)`; the layers are yours to choose. `forward(x, n_stages=None)` returns `(x_hat, vq_loss, codes)` with codes `(B, n_used, L / hop_length)`, and raises `ValueError` unless `x` is `(B, 1, L)` with `L % hop_length == 0`. `encode(x, n_stages=None)` returns the codes; `decode(codes)` the waveform, so that `decode(encode(x, n))` equals `forward(x, n)[0]`. `multiscale_spectral_loss(x, x_hat, *, n_ffts=(64, 128, 256, 512), eps=1e-5)`: for each `n` in `n_ffts`, $S = $ `torch.stft(x, n, n // 4, window=torch.hann_window(n), center=True, pad_mode="constant", return_complex=True).abs()` (the same STFT as your topic 24 one), and the term `mean|S - S_hat| + mean|log(S + eps) - log(S_hat + eps)|`; returns the mean over scales. `x` may be `(B, 1, L)` or `(B, L)`.

> **🔮 predict:** your codec at 2 kbps against 4-bit μ-law at 32 kbps, sixteen times the rate. which has the higher SNR? the lower log-spectral distance?

> **▶ build: stage 5.** implement `ConvCodec` and `multiscale_spectral_loss`.

## worked example

**RVQ by hand.** codebooks $C^{(1)} = \{(0, 0), (4, 0)\}$ and $C^{(2)} = \{(1, 0), (0, 1), (-1, 0)\}$; encode $x = (5, 0.9)$. stage 1: $(4, 0)$ is nearer (squared distance $1.81$ against $25.81$), code 1, residual $(1, 0.9)$. stage 2: distances to the residual are $0.81$, $1.01$ and $4.81$, so code 0, $(1, 0)$, residual $(0, 0.9)$. codes $(1, 0)$, reconstruction $(5, 0)$, error $0.81$ in squared norm; decoding stage 1 alone gives $(4, 0)$ with error $1.81$. three bits ($1 + \log_2 3 \approx 2.58$, rounded up per stage as $1 + 2$), five codewords.

**bits.** at 8 kHz with hop 32 (strides 2, 4, 4), 250 frames/s; 8 stages of 256 codes is $250\cdot8\cdot8 = 16$ kbps. the same codec with strides (2, 4) and 64 codes per stage: 1,000 frames/s × 6 bits = 6 kbps per stage. 2-bit midrise PCM has levels $-0.75, -0.25, 0.25, 0.75$ and costs 16 kbps at 8 kHz; an input of 0.3 becomes 0.25, an input of 1.7 is clipped to 1 and becomes 0.75.

your `rvq_encode`, `rvq_decode`, `bitrate` and `pcm_quantize` must reproduce these.

## on paper

1. **codebook sizes.** EnCodec at 24 kbps sends 32 codes of 10 bits per frame. how many codewords would a single VQ need for the same rate, and how many does the RVQ store?

<details><summary>solution</summary>

320 bits per frame: a single codebook of $2^{320} \approx 2\times10^{96}$ codewords, more than there are atoms in the observable universe. the RVQ stores $32\times1024 = 32{,}768$. the exponential becomes linear because the reachable reconstructions are sums of one codeword per stage rather than arbitrary points.

</details>

2. **6 dB per bit.** a signal uniform on $[-1, 1]$ is quantized by a $b$-bit midrise quantizer. show that the SNR is $6.02b$ dB.

<details><summary>solution</summary>

the step is $\Delta = 2/2^b$, and the error is uniform on $[-\Delta/2, \Delta/2]$ with variance $\Delta^2/12$. the signal's power is $1/3$. so $\text{SNR} = (1/3)/(\Delta^2/12) = 4/\Delta^2 = 2^{2b}$, i.e. $10\log_{10}2^{2b} = 6.02b$ dB. each bit quarters the error. real signals don't fill $[-1, 1]$ uniformly; speech mostly sits at low amplitudes, which is why μ-law helps.

</details>

3. **one straight-through, not $S$.** suppose each stage outputs $r_{s-1} + \operatorname{sg}(q_s - r_{s-1})$ and you sum them, with $r_s = r_{s-1} - \operatorname{sg}(q_s)$. what is $\partial z_q/\partial z_e$? compare with $z_e + \operatorname{sg}(q - z_e)$.

<details><summary>solution</summary>

each $r_{s-1} = z_e - \sum_{j<s}\operatorname{sg}(q_j)$ has Jacobian $I$ with respect to $z_e$, and so does each stage's output, so the sum has Jacobian $SI$: the encoder's gradient is multiplied by the number of stages (and, with quantizer dropout, by a random number). the single straight-through $z_e + \operatorname{sg}(q - z_e)$ has Jacobian exactly $I$, and the same forward value $q$.

</details>

4. **SNR against LSD.** a codec outputs the input delayed by one sample. for a 1 kHz tone at 8 kHz, what are the SNR and (roughly) the LSD?

<details><summary>solution</summary>

the error is $\sin(\omega n) - \sin(\omega(n - 1))$ with $\omega = 2\pi/8$, an amplitude of $2\sin(\omega/2) = 0.765$, so SNR $= -20\log_{10}0.765 = 2.3$ dB: "terrible". the delayed signal has the same magnitude spectrum in every frame (up to edge effects), so the LSD is near 0 dB. and a one-sample delay is inaudible. SNR measures waveform fidelity; it's the wrong yardstick for codecs that are allowed to change phase.

</details>

## experiment

experiment 1 compares RVQ with a single codebook at the same number of bits, on 8-sample frames of synthetic audio, and measures how much each stage buys. experiment 2 is the block's **project**: train your `ConvCodec` with quantizer dropout on synthetic audio, then map bitrate against SNR and log-spectral distance for 1 to 8 stages (2–16 kbps), next to PCM, μ-law and "k-means RVQ on raw waveform frames", a codec with no neural network at all. experiment 3 measures how predictable the codec's tokens are: their entropy, and what a simple context model could save.

## check your understanding

1. why is RVQ greedy encoding not guaranteed to find the best code tuple, and why do codecs use it anyway?

<details><summary>answer</summary>

stage 1 picks the codeword nearest to $x$ without knowing what later stages can correct; a slightly worse first choice could leave a residual that stage 2 represents almost exactly. searching all $K^S$ tuples is infeasible, and beam searches over tuples help only a little. greedy encoding is $O(SK)$ per vector and keeps the coarse-to-fine ordering that makes dropping stages meaningful.

</details>

2. why does quantizer dropout make one model serve many bitrates?

<details><summary>answer</summary>

without it, the decoder only ever sees the full sum of $S$ code vectors, and an input made from the first few stages is out of distribution for it. with dropout, it's trained on every prefix length, so it learns to decode coarse codes well and to use the finer stages as refinements. SoundStream reports that one model trained this way nearly matches separate fixed-bitrate models.

</details>

3. why might a codec with a *lower* SNR sound *better*?

<details><summary>answer</summary>

SNR penalizes every waveform difference equally, including phase shifts and slight time misalignments the ear can't hear, and it rewards smooth, averaged reconstructions that sound muffled. a decoder that synthesizes plausible detail with the right spectral envelope (what adversarial training encourages) can have worse SNR and sound far better. that's why the neural codec papers report listening tests.

</details>

4. the entropy of a stage's codes is 5.1 bits with 256 codes. what does that tell you about the bitrate, and what doesn't it tell you?

<details><summary>answer</summary>

a lossless entropy coder treating codes as independent draws could send that stage at about 5.1 instead of 8 bits per code, a 36% saving, with no loss. it doesn't account for dependencies between codes (across time or across stages), which a context model or language model could exploit to save more; and a low entropy can also mean wasted capacity, many codewords that are rarely or never used.

</details>

5. why are RVQ tokens convenient for audio language models, compared with, say, quantized mel spectrograms?

<details><summary>answer</summary>

they come with a decoder that turns them back into a waveform at high quality, they're compact (dozens to hundreds of tokens per second instead of thousands of samples), and the coarse-to-fine stage order lets a model generate the coarse stages first and condition the fine ones on them. a quantized mel spectrogram would still need a vocoder and has no such ordering.

</details>

## common bugs

- **quantizing $x$ at every stage instead of the residual.** every stage then picks roughly the same codeword and the error stops falling after stage 1.
- **summing per-stage straight-through outputs.** the encoder's gradient is scaled by the number of stages used, which changes from batch to batch under quantizer dropout.
- **detaching the stage losses.** the codebooks then never move, and only the k-means initialization does any work.
- **bitrate without the log.** $K$ codes cost $\log_2 K$ bits, not $K$.
- **reshaping `(B, D, T)` straight to `(-1, D)`.** that mixes time and channels; move the channel axis last first (`movedim(1, -1)`).
- **comparing codecs at different bitrates, or on one metric.** a curve at matched rates, on both SNR and LSD, with a clear statement of what neither measures.

## going deeper

- Zeghidour, Luebs, Omran, Skoglund & Tagliasacchi (2021), *SoundStream: an end-to-end neural audio codec*. RVQ, quantizer dropout, adversarial and spectral losses. [arXiv:2107.03312](https://arxiv.org/abs/2107.03312)
- Défossez, Copet, Synnaeve & Adi (2022), *High fidelity neural audio compression* (EnCodec). k-means initialization, a loss balancer, and a transformer entropy model over the codes. [arXiv:2210.13438](https://arxiv.org/abs/2210.13438)
- Juang & Gray (1982), *Multiple stage vector quantization for speech coding*, ICASSP. the original residual (multi-stage) VQ. [doi:10.1109/ICASSP.1982.1171604](https://doi.org/10.1109/ICASSP.1982.1171604)
- van den Oord, Vinyals & Kavukcuoglu (2017), *Neural discrete representation learning* (VQ-VAE). [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
- Borsos et al. (2022), *AudioLM: a language modeling approach to audio generation*; Wang et al. (2023), *Neural codec language models are zero-shot text to speech synthesizers* (VALL-E). [arXiv:2209.03143](https://arxiv.org/abs/2209.03143), [arXiv:2301.02111](https://arxiv.org/abs/2301.02111)
- Rix, Beerends, Hollier & Hekstra (2001), *Perceptual evaluation of speech quality (PESQ)*, ICASSP; Hines, Skoglund, Kokaram & Harte (2015), *ViSQOL: an objective speech quality model*, EURASIP Journal on Audio, Speech, and Music Processing. the perceptual metrics this topic can't run.

## connections

- ← builds on: `VectorQuantizer`, `codebook_usage` and `restart_dead_codes` (ch 21 · topic 04); k-means++ and Lloyd's algorithm (ch 11 · topic 01); entropy (ch 05); strided and transposed convolutions (ch 16); your STFT (topic 24).
- → used in: the elective portfolio (chapter 29 capstone) takes this project's per-seed numbers; codec tokens are the input format of audio language models, which are GPTs (ch 18) over RVQ codes. after this topic you may use library RVQ implementations (e.g. `vector-quantize-pytorch`) and pretrained EnCodec models for audio tokenization, since you've built both pieces.
