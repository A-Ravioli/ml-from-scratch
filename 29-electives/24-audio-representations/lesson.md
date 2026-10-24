# audio representations

## why this matters

a microphone hands you 16,000 numbers per second, and almost none of them, taken alone, tell you anything. say "ah" twice and the two waveforms won't line up sample for sample: the pitch drifts, the harmonics' phases differ, the loudness changes. what stays the same is the *shape of the spectrum over a few tens of milliseconds*: which bands are loud. that is what your ear tracks, and what speech recognizers and audio codecs are built on.

getting from samples to that shape is a chain of tradeoffs. what happens to frequencies too high for your sample rate? how long a stretch do you analyze at once, when long stretches sharpen frequency but blur timing? how do you space the frequency bands? how do you get a short vector that keeps the vowel but ignores the pitch? this topic builds the chain: the DFT, the short-time Fourier transform and its exact inverse, the mel filterbank, and MFCCs. topic 25 trains a speech recognizer on these features; topic 26 judges a neural codec with them.

**all audio in this block is synthetic.** there's no speech dataset in the course cache and nothing gets downloaded, so every sound is made by a few lines of NumPy at the bottom of the exercise notebook: tones, chirps, and a source–filter model of sung vowels. that's enough to see every effect in this topic, and not enough to claim anything about real recordings.

## warm-up: predict first

1. a 7 kHz tone is sampled at 8 kHz. what frequency does the sampled signal contain?
2. you analyze 16 kHz audio with 512-sample frames. how far apart are the DFT's bins? could you tell 1,000 Hz from 1,020 Hz?
3. at 8 kHz you want to separate two tones 40 Hz apart *and* two clicks 6 ms apart. is there a frame length that does both?
4. you halve the hop between frames. does that improve frequency resolution? time resolution?
5. you sing /a/ at 100 Hz, then at 140 Hz, then /i/ at 100 Hz. in a log spectrum, which pair is further apart: the two /a/s, or /a/ and /i/?

## prerequisites check

- [ ] you can project a function onto a Fourier basis, and know sines of different frequencies are orthogonal (→ ch 03 · topic 07)
- [ ] complex exponentials $e^{i\theta} = \cos\theta + i\sin\theta$; matrix–vector products as changes of basis (→ ch 02)
- [ ] 1-D convolution and stride (→ ch 16 · topic 01): a spectrogram is a bank of strided filters
- [ ] least squares and its normal equations (→ ch 02 · topic 04)
- [ ] decibels: $10\log_{10}$ of a power ratio, so ×10 in power is +10 dB and ×10 in amplitude is +20 dB

## core ideas

### idea 1: sampling, aliasing and the DFT

**sampling.** a digital signal is $x[n] = x(n/f_s)$, the continuous signal read every $1/f_s$ seconds (8 kHz for telephone speech, 16 kHz for most speech recognition, 44.1 kHz for CDs). sampling loses information in a precise way: a cosine at $f$ Hz and one at $f + f_s$ Hz give *identical* samples, since $\cos(2\pi(f + f_s)n/f_s) = \cos(2\pi fn/f_s + 2\pi n)$, and so do $f$ and $f_s - f$, since cosine is even. every frequency gets folded into $[0, f_s/2]$: subtract the nearest multiple of $f_s$ and take the absolute value. the top of that band, $f_s/2$, is the **Nyquist frequency**. a frequency above it doesn't vanish; it reappears below it as an **alias**, indistinguishable from a real component there. that's why audio interfaces low-pass filter before sampling.

**the DFT.** the discrete Fourier transform writes $N$ samples in a basis of $N$ complex exponentials:

$$X[k] = \sum_{n=0}^{N-1} x[n]\,e^{-2\pi i kn/N}, \qquad k = 0, \dots, N - 1.$$

bin $k$ measures the frequency $kf_s/N$ Hz: $|X[k]|$ is its amplitude, $\arg X[k]$ its phase. the bins are $f_s/N$ Hz apart, set by the frame's duration $N/f_s$ and nothing else. as with Fourier projections in ch 03, the vectors $e_k[n] = e^{2\pi ikn/N}$ are orthogonal ($\|e_k\|^2 = N$), so the DFT is $N$ inner products: a product with the **DFT matrix** $W_{kn} = e^{-2\pi ikn/N}$, inverted by $x = \frac1N W^*X$. for real $x$, $X[N - k] = \overline{X[k]}$, so bins $0..\lfloor N/2\rfloor$ carry everything; that's `np.fft.rfft`. the FFT computes the same numbers in $O(N\log N)$ instead of $O(N^2)$. once you've built the matrix version, use `np.fft`.

> **spec.** `alias_frequency(f, sr)` returns the frequency in $[0, sr/2]$ at which a real sinusoid of `f` Hz appears after sampling at `sr` Hz: a float for scalar input, elementwise for arrays. `dft(x)` returns the DFT of `x` along its last axis (any leading shape, real or complex) as $xW^\top$; it must not call `np.fft`, and equals `np.fft.fft(x, axis=-1)` (unnormalized, minus sign in the exponent).

> **▶ build: stage 1.** implement `alias_frequency` and `dft`.

### idea 2: windows, frames and the STFT

speech changes every few tens of milliseconds, so one DFT of a whole utterance mixes every sound in it. instead, cut the signal into overlapping **frames**, frame $m$ starting at sample $mH$ for a **hop** $H$, and take a DFT of each. side by side, they form the **short-time Fourier transform**

$$X[k, m] = \sum_{n=0}^{N-1} w[n]\,x[mH + n]\,e^{-2\pi i kn/N},$$

and $|X[k, m]|^2$ is the **spectrogram**. the **window** $w$ tapers each frame to zero at its edges. with a rectangular window, the frame's abrupt ends leak a tone's energy into every bin, with sidelobes only 13 dB below the peak. the **Hann window** $w[n] = \tfrac12 - \tfrac12\cos(2\pi n/N)$ pushes the first sidelobe to −31 dB, and pays with a main lobe twice as wide: a pure tone now spans 4 bins between nulls instead of 2. dividing by $N$ gives the **periodic** Hann window, one full cosine period, whose copies shifted by $N/2$ add up exactly (idea 3). the **symmetric** version divides by $N - 1$ so both ends are 0; it's meant for filter design, not spectral analysis.

the STFT is also a filter bank: for each $k$, $X[k, \cdot]$ is the signal convolved with a modulated window $w[n]e^{2\pi ikn/N}$ (a band-pass filter at bin $k$), subsampled with stride $H$: $N/2 + 1$ band-pass filters of equal width.

**the time–frequency tradeoff.** a frame lasts $N/f_s$ seconds and its bins are $f_s/N$ Hz apart, so time resolution times frequency resolution is fixed. with Hann, two tones closer than about two bins merge into one peak; two events closer than about a frame merge into one blob. this is the uncertainty relation of Gabor (1946). two tempting fixes don't beat it: a smaller hop gives more frames, but each still averages over $N$ samples; and **zero-padding** a frame to a longer DFT samples the same blurred spectrum on a finer grid without narrowing any peak.

**centering.** with `center=True` (the convention of librosa and `torch.stft`) you pad $N/2$ samples on both sides first, so frame $m$ is centred on sample $mH$. this course pads with zeros, as `scipy.signal.stft(..., boundary="zeros")` does; libraries that pad differently disagree only in the edge frames.

> **spec.** `hann(n, *, periodic=True)`: $\tfrac12 - \tfrac12\cos(2\pi k/D)$, $k = 0..n-1$, with $D = n$ (periodic) or $n - 1$ (symmetric); `hann(1)` is `[1.0]`. `frame(x, frame_length, hop_length)` returns the `(n_frames, frame_length)` array of `x[m*hop : m*hop + frame_length]`, `n_frames = 1 + (len(x) - frame_length) // hop_length` (an incomplete last frame is dropped); `ValueError` if `x` is shorter than one frame. `stft(x, n_fft, hop_length, *, window=None, center=True)` returns the complex, *unscaled* `(n_fft // 2 + 1, n_frames)` array $X[k, m]$ (frequency first): `window` defaults to `hann(n_fft)`; `center=True` pads `n_fft // 2` zeros on both sides. it equals `scipy.signal.stft(x, window='hann', nperseg=n_fft, noverlap=n_fft - hop, boundary='zeros', padded=False)` times $\sum w$ (scipy scales by $1/\sum w$).

> **🔮 predict:** at $f_s = 8$ kHz, two tones 40 Hz apart. what Hann frame length $N$ separates them into two peaks? is it the $N$ whose bin spacing is 40 Hz?

> **▶ build: stage 2.** implement `hann`, `frame` and `stft`.

### idea 3: overlap-add and the inverse STFT

a codec or a denoiser edits a spectrogram and then needs a waveform back. with $H < N$ every sample sits in several frames, and inverting means combining those copies.

the inverse DFT of frame $m$ returns $y_m[n] = w[n]\,x[mH + n]$. **overlap-add (OLA)** adds the segments back at their positions, giving $x[n]\sum_m w[n - mH]$. that's $x$ (up to a constant) when the shifted windows sum to a constant: the **constant overlap-add (COLA)** condition. the periodic Hann window is COLA at hops $N/2, N/4, \dots$ because a half-period shift flips the cosine's sign: $w[n] + w[n + N/2] = 1$. the symmetric Hann isn't COLA at $N/2$, one more reason to use the periodic one.

the better inverse applies the window again (a **synthesis window**) before overlap-add and divides by the overlap-added *squared* window:

$$\hat x[n] = \frac{\sum_m w[n - mH]\,y_m[n - mH]}{\sum_m w^2[n - mH]}.$$

for an unmodified STFT the numerator is $x[n]\sum_m w^2[n - mH]$, so $\hat x = x$ exactly for *any* window and hop that keep the denominator nonzero, COLA or not. for a *modified* STFT (a mask, a quantized codec output) there's usually no signal with exactly that STFT, and Griffin & Lim (1984) showed this formula gives the signal whose STFT is closest in squared error (on paper 3). it's the inverse in `librosa`, `torch.istft` and `scipy.signal.istft`. COLA still helps: it keeps the denominator constant, so edits aren't reweighted unevenly across time.

> **spec.** `is_cola(window, hop_length, *, tol=1e-10)` returns a Python bool: whether copies of `window` shifted by every multiple of `hop_length` sum to the same value at every sample (relative tolerance `tol`). `istft(X, hop_length, *, n_fft=None, window=None, center=True, length=None)`: `n_fft` defaults to `len(window)` if a window is given, else `2 * (X.shape[0] - 1)` (pass odd sizes explicitly); `window` defaults to `hann(n_fft)`. inverse-rfft each column (`n=n_fft`), multiply by the window, overlap-add at `hop_length`, divide by the overlap-added squared window wherever it exceeds `1e-10` times its maximum (0 elsewhere). with `center=True` drop the first `n_fft // 2` samples, and, if `length` is None, the last `n_fft // 2` too. a given `length` truncates or zero-pads to exactly that many samples.

> **▶ build: stage 3.** implement `is_cola` and `istft`.

### idea 4: the mel scale

the STFT's bins are evenly spaced in Hz; hearing isn't. you easily tell 200 Hz from 250 Hz, but 8,000 Hz and 8,050 Hz sound the same: above about 1 kHz, pitch perception is roughly logarithmic. the **mel scale** (Stevens, Volkmann & Newman, 1937) measures frequency in perceptually equal steps. there's no single mel formula, so know which one a library uses. this course uses the **HTK formula**

$$m(f) = 2595\log_{10}\Big(1 + \frac{f}{700}\Big), \qquad f(m) = 700\big(10^{m/2595} - 1\big),$$

roughly linear below 700 Hz and logarithmic above, with $m(1000) \approx 1000$. librosa's default **Slaney formula** is exactly linear below 1 kHz, so its band edges differ.

a **mel filterbank** groups the bins into $M$ bands evenly spaced in mel. take $M + 2$ points equally spaced in mel from $f_{\min}$ to $f_{\max}$ (default $f_s/2$), convert them to Hz, $f_0 < \dots < f_{M+1}$, and let filter $j$ be a triangle rising linearly in Hz from 0 at $f_{j-1}$ to 1 at $f_j$ and back to 0 at $f_{j+1}$, evaluated at the bin frequencies $kf_s/N$. neighbours overlap by half and, between the first and last centres, sum to exactly 1. low filters span a few bins, high ones dozens. this course uses **peak height 1**, as HTK does (`librosa.filters.mel(..., htk=True, norm=None)`); librosa's default `norm="slaney"` scales each triangle to unit area, shrinking the wide high bands. a real difference, not a rounding detail.

the **mel spectrogram** is $\text{Mel}[j, m] = \sum_k F[j, k]\,|X[k, m]|^2$: from $N/2 + 1$ numbers per frame down to $M$ (typically 40–128). loudness is perceived roughly logarithmically, so features are taken in decibels, $10\log_{10}\max(S, a_{\min})$. the floor $a_{\min}$ keeps silence finite; an optional `top_db` clips everything more than that far below the loudest value.

> **spec.** `hz_to_mel(f)`, `mel_to_hz(m)`: the HTK formulas, elementwise. `mel_filterbank(sr, n_fft, n_mels, *, fmin=0.0, fmax=None)`: the `(n_mels, n_fft // 2 + 1)` triangles above (peak 1, no area normalization), bin frequencies `k * sr / n_fft`, `fmax` defaulting to `sr / 2`. `mel_spectrogram(x, sr, *, n_fft=512, hop_length=128, n_mels=40, fmin=0.0, fmax=None, power=2.0)` returns `mel_filterbank(...) @ np.abs(stft(x, n_fft, hop_length)) ** power`. `power_to_db(S, *, ref=1.0, amin=1e-10, top_db=80.0)` returns `10 * log10(max(S, amin) / ref)`, then, unless `top_db` is None, raises everything below `max - top_db` to `max - top_db`.

> **▶ build: stage 4.** implement the five functions of stage 4.

### idea 5: MFCCs: the envelope without the pitch

voiced speech fits a **source–filter model** (Fant, 1960): the vocal folds buzz at the pitch $f_0$, a comb of harmonics at $f_0, 2f_0, \dots$, and the vocal tract filters the buzz, boosting a few resonant bands called **formants**. the vowel is the filter: /i/ has its first two formants near 270 and 2,290 Hz, /a/ near 730 and 1,090 Hz (Peterson & Barney, 1952). the pitch is the source: you can sing /a/ on any note. the notebook's `harmonic_vowel` is exactly this model.

in the spectrum, source and filter multiply, $|X(f)| = |S(f)|\,|H(f)|$; in the log they add. across frequency, $\log|H|$ varies slowly (a few broad bumps), while $\log|S|$ ripples once every $f_0$ Hz. so a second transform, *across frequency*, separates them: slow variation lands in the first coefficients, fast ripple in the high ones. a transform of a log spectrum is a **cepstrum**.

**MFCCs** (mel-frequency cepstral coefficients; Davis & Mermelstein, 1980) apply a **DCT-II** across the log mel bands of each frame,

$$c_q = s_q\sum_{j=0}^{M-1}\log\text{Mel}_j\,\cos\Big(\frac{\pi q(2j + 1)}{2M}\Big), \qquad s_0 = \sqrt{1/M},\ s_{q>0} = \sqrt{2/M},$$

and keep the first 13 or so. the mel filters already smear out much of the harmonic ripple (each band above the lowest few spans several harmonics) and truncation discards more. with this scaling the DCT is orthonormal, so it preserves distances, and $c_0$ is $\sqrt M$ times the mean log energy: a gain change moves only $c_0$ (on paper 4). the DCT also roughly decorrelates the bands, which suited the diagonal-covariance Gaussians of classical recognizers.

> **spec.** `dct2(x, *, axis=0)`: the orthonormal DCT-II above along `axis`, equal to `scipy.fft.dct(x, type=2, norm='ortho', axis=axis)` (build the matrix once to see the scaling). `mfcc(x, sr, *, n_mfcc=13, n_fft=512, hop_length=128, n_mels=40, fmin=0.0, fmax=None)` returns the `(n_mfcc, n_frames)` array `dct2(power_to_db(mel_spectrogram(x, sr, ...), top_db=None), axis=0)[:n_mfcc]`, passing every keyword through. no `top_db` clipping, so each frame's coefficients depend only on that frame.

> **🔮 predict:** you multiply a signal by 10. which MFCCs change, and by how much? (40 mel bands.)

> **▶ build: stage 5.** implement `dct2` and `mfcc`.

## worked example

**aliasing.** at $f_s = 8$ kHz, a 7 kHz tone appears at $|7000 - 8000| = 1000$ Hz; a 12.5 kHz tone at $|12500 - 16000| = 3500$ Hz; a 4 kHz tone sits at Nyquist and stays put.

**a 4-point DFT.** $x = (1, 0, -1, 0)$ is one cycle of a cosine. $X[k] = 1 - e^{-i\pi k}$, so $X = (0, 2, 0, 2)$: bin 1 and its mirror, bin 3, each with magnitude $N/2$.

**COLA.** the periodic Hann window with $N = 4$ is $(0, \tfrac12, 1, \tfrac12)$. at hop 2, positions 0 and 2 overlap ($0 + 1$), as do 1 and 3 ($\tfrac12 + \tfrac12$): constant, so COLA. at hop 3, the three positions collect $w[0] + w[3] = \tfrac12$, $w[1] = \tfrac12$ and $w[2] = 1$: not COLA.

**one mel filter.** $f_s = 8000$, $N = 8$ (bins at 0, 1000, …, 4000 Hz), one filter from 0 to 4,000 Hz. $m(4000) = 2146.1$ mel, so the edges are 0, 1073.0 and 2146.1 mel, i.e. 0, $700(10^{1073.0/2595} - 1) = 1113.9$ and 4,000 Hz. the centre sits at 1,114 Hz, not 2,000 Hz: mel crowds filters toward low frequencies. the weights are

| bin | 0 Hz | 1000 Hz | 2000 Hz | 3000 Hz | 4000 Hz |
|---|---|---|---|---|---|
| weight | 0 | $1000/1113.9 = 0.898$ | $2000/2886.1 = 0.693$ | $1000/2886.1 = 0.346$ | 0 |

your `alias_frequency`, `dft` and `mel_filterbank(8000, 8, 1)` must reproduce these numbers.

## on paper

1. **aliasing.** show that cosines at $f$ and at $|f - kf_s|$ Hz ($k$ an integer) have the same samples, and conclude that the apparent frequency $|f - f_s\operatorname{round}(f/f_s)|$ lies in $[0, f_s/2]$.

<details><summary>solution</summary>

$\cos(2\pi(f - kf_s)n/f_s) = \cos(2\pi fn/f_s - 2\pi kn) = \cos(2\pi fn/f_s)$, and cosine is even, so the sign of $f - kf_s$ doesn't matter. the smallest $|f - kf_s|$ is at the integer $k$ nearest $f/f_s$, where $|f/f_s - k| \le \tfrac12$.

</details>

2. **COLA for the periodic Hann window.** show $w[n] + w[n + N/2] = 1$, then show $\sum_m w^2[n - mN/4] = 3/2$ at hop $N/4$.

<details><summary>solution</summary>

$w[n + N/2] = \tfrac12 - \tfrac12\cos(\theta + \pi) = \tfrac12 + \tfrac12\cos\theta$ with $\theta = 2\pi n/N$, so the pair sums to 1. for the squares, $w^2 = \tfrac38 - \tfrac12\cos\theta + \tfrac18\cos 2\theta$. at hop $N/4$ four copies overlap, shifted in $\theta$ by $0, \tfrac\pi2, \pi, \tfrac{3\pi}2$: their $\cos\theta$ terms cancel, and so do their $\cos 2\theta$ terms (shifts $0, \pi, 2\pi, 3\pi$ alternate in sign), leaving $4\cdot\tfrac38 = \tfrac32$. so here $\sum w = 2$ but $\sum w^2 = 1.5$: divide by the wrong one and everything comes out scaled.

</details>

3. **the least-squares inverse.** given target segments $y_m$, find $\hat x$ minimizing $\sum_m\sum_n(w[n]\hat x[mH + n] - y_m[n])^2$ (by Parseval, the squared distance between $\hat x$'s STFT and the target, up to a factor $N$).

<details><summary>solution</summary>

sample $\hat x[t]$ appears only in terms with $mH + n = t$, so the objective splits into one 1-D quadratic per sample, $\sum_m(w[t - mH]\hat x[t] - y_m[t - mH])^2$. its minimizer is $\hat x[t] = \sum_m w[t - mH]\,y_m[t - mH]/\sum_m w^2[t - mH]$, the formula of idea 3. the checks compare your `istft` with an explicit `np.linalg.lstsq` solve.

</details>

4. **gain and $c_0$.** show that scaling a signal by $g$ changes only $c_0$, by $20\log_{10}(g)\sqrt M$.

<details><summary>solution</summary>

power scales by $g^2$, the filterbank is linear, so every mel band gains $\delta = 20\log_{10}g$ dB (if nothing sits at the floor). the DCT is linear, so the coefficients change by the DCT of $\delta\mathbf 1$: zero for $q \ge 1$ (cosines orthogonal to constants) and $\sqrt{1/M}\cdot M\delta = \sqrt M\delta$ for $q = 0$. with $M = 40$, $g = 10$: $+126.5$.

</details>

## experiment

experiment 1 analyzes two tones 40 Hz apart and two clicks 6 ms apart at frame lengths from 64 to 1,024 samples, and checks which frame lengths resolve which (and whether zero-padding helps). experiment 2 sweeps a chirp up to twice the Nyquist frequency and overlays your `alias_frequency` on its spectrogram. experiment 3 asks what moves the features more, a change of pitch or a change of vowel, in the log spectrum, the log mel spectrum and MFCCs.

## check your understanding

1. why low-pass filter *before* sampling rather than after?

<details><summary>answer</summary>

after sampling, an alias is indistinguishable from a genuine component at the folded frequency: a 7 kHz tone sampled at 8 kHz gives exactly the numbers a 1 kHz tone would. the high frequencies must go while they're still distinguishable: in the analogue signal, or, when downsampling digitally, before dropping samples.

</details>

2. zero-padding a 256-sample frame to a 2,048-point DFT gives 8× more bins. why doesn't it separate two tones the 256-sample frame couldn't?

<details><summary>answer</summary>

it samples the same continuous spectrum (that of the windowed 256-sample frame) at more points. each peak's width is set by the window's length, about $4f_s/256$ for Hann, and zero-padding doesn't change the window. you see the blurred peak in more detail; it doesn't get narrower.

</details>

3. why does the inverse STFT divide by $\sum w^2$ rather than $\sum w$?

<details><summary>answer</summary>

the window is applied twice, in analysis and again as a synthesis window, so each sample's total weight is $\sum w^2$. the synthesis window is what makes the result least-squares optimal for modified STFTs, and it tapers edited frames instead of leaving discontinuities. dividing by $\sum w$ is right only for plain overlap-add without a synthesis window, and then exact only for unmodified STFTs.

</details>

4. a model trained on librosa's default mel spectrograms is fed your `mel_spectrogram`. what differs?

<details><summary>answer</summary>

the mel formula (Slaney's, linear below 1 kHz, vs HTK's) moves the band edges; librosa's default area normalization shrinks the high bands relative to peak-1 triangles; and its defaults are a 2048-point FFT and 128 mels. the model would see systematically different numbers. always check which conventions a pretrained model expects.

</details>

5. why keep only the first 13 MFCCs?

<details><summary>answer</summary>

low coefficients describe the slowly varying envelope (the formants, which identify the vowel); high ones describe fast ripple across frequency, mostly harmonic structure and noise. truncating keeps more of *what* was said and less of the pitch it was said on. 13 is a convention from classical recognizers, not a law; the lab measures what the mel step and the truncation each buy.

</details>

6. MFCCs ignore phase. what does that cost a recognizer, and what does it cost a codec?

<details><summary>answer</summary>

a recognizer, very little: phoneme identity is largely insensitive to the relative phases of harmonics within a frame. a codec, everything: you can't resynthesize a waveform from magnitudes without estimating phase (Griffin & Lim's iterative algorithm does, with audible artifacts). that's why topic 26's codec works on waveforms and uses spectral features only to *judge* quality.

</details>

## common bugs

- **a transposed STFT.** shapes `(n_frames, n_bins)` make the filterbank product fail, or silently use the wrong axis when the sizes match. the convention here (librosa's) is frequency first.
- **reflect padding, or none, when the spec says zeros.** only the edge frames differ from scipy, so a check on middle frames passes and a full comparison fails.
- **the symmetric Hann window for analysis.** the STFT is off everywhere by a little, and overlap-add at hop $N/2$ ripples.
- **dividing by $\sum w$ in the inverse.** reconstruction is off by a constant (1.33 for Hann at hop $N/4$), or ripples where $\sum w$ is constant but $\sum w^2$ isn't.
- **$M$ mel edge points instead of $M + 2$, or spacing in Hz.** the worked example's centre lands at 2,000 Hz instead of 1,114 Hz.
- **the log of zero.** digital silence gives `-inf`, then NaN after the DCT. floor at `amin` first.

## going deeper

- Davis & Mermelstein (1980), *Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences*, IEEE Transactions on Acoustics, Speech, and Signal Processing 28(4). the paper that made MFCCs standard. [doi:10.1109/TASSP.1980.1163420](https://doi.org/10.1109/TASSP.1980.1163420)
- Griffin & Lim (1984), *Signal estimation from modified short-time Fourier transform*, IEEE TASSP 32(2). the least-squares inverse and iterative phase retrieval. [doi:10.1109/TASSP.1984.1164317](https://doi.org/10.1109/TASSP.1984.1164317)
- Harris (1978), *On the use of windows for harmonic analysis with the discrete Fourier transform*, Proceedings of the IEEE 66(1). windows, main lobes and sidelobes. [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837)
- Stevens, Volkmann & Newman (1937), *A scale for the measurement of the psychological magnitude pitch*, JASA 8(3). [doi:10.1121/1.1915893](https://doi.org/10.1121/1.1915893)
- Peterson & Barney (1952), *Control methods used in a study of the vowels*, JASA 24(2). the formants behind `vowel_formants()`. [doi:10.1121/1.1906875](https://doi.org/10.1121/1.1906875)
- Gabor (1946), *Theory of communication*, Journal of the IEE 93(26); Fant (1960), *Acoustic Theory of Speech Production*, Mouton.

## connections

- ← builds on: Fourier projections and orthogonal bases (ch 03 · topic 07); changes of basis and least squares (ch 02); strided convolution, for the filter-bank view (ch 16).
- → used in: topic 25 trains a CTC recognizer on your log mel spectrograms of synthetic phoneme sequences; topic 26 judges a neural codec with a log-spectral distance built on your `stft`. after this topic you may use `np.fft`, `scipy.signal.stft`/`istft` and `torch.stft`/`torch.istft`.
