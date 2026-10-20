# audio representations · lab notes

numbers are from the reference solution; the lab takes about 2 s. every signal is synthetic, so nothing here varies with a dataset. the only randomness is the harmonic phases and the noise in experiment 3, which uses three seeds.

## experiment 1: the time–frequency tradeoff

**what people usually predict:** the tones separate once the bins are 40 Hz apart or closer, i.e. $N = 8000/40 = 200$, so 256 samples; zero-padding to 2,048 points (3.9 Hz bins) certainly separates them; and a window that resolves the tones can probably resolve clicks 6 ms apart too, "with a small hop".

**what happens:**

| frame length N | duration | bin spacing | tones 40 Hz apart resolved? | clicks 6 ms apart resolved? |
|---|---|---|---|---|
| 64 | 8 ms | 125.0 Hz | no | yes |
| 128 | 16 ms | 62.5 Hz | no | no |
| 256 | 32 ms | 31.2 Hz | no | no |
| 384 | 48 ms | 20.8 Hz | yes | no |
| 512 | 64 ms | 15.6 Hz | yes | no |
| 1024 | 128 ms | 7.8 Hz | yes | no |
| 256 zero-padded to 2048 | 32 ms | 3.9 Hz | no | – |

the tones need $N \ge 384$: a bin spacing of about half the tone separation, not equal to it. the clicks need $N \le 64$, about one window per 6 ms. no frame length does both, and the hop doesn't help: the click test already uses a hop of 8 samples (1 ms). zero-padding gives eight times more bins and still one peak.

**why:** a Hann window's main lobe spans 4 bins between nulls, so each tone smears over about ±2 bins; two lobes 1.3 bins apart (N = 256) overlap into one hump, and they only develop a 3 dB dip between them once they're about 2 bins apart. in time, a frame of $N$ samples responds to a click for its whole duration (the right-hand plot shows the energy envelope of each click as wide as the window), so two clicks merge once the window is much longer than their spacing. the tone condition is $N/f_s \gtrsim 2/\Delta f = 50$ ms and the click condition $N/f_s \lesssim 8$ ms: to do both, a window would need (duration × 40 Hz) below 0.32, while separating the tones needs about 2. zero-padding evaluates the same 32 ms window's spectrum at more frequencies: the lobes are just as wide, only drawn more smoothly. one more thing worth noticing in the spectrogram at $N = 64$: the two tones appear as a *single* band at 1,020 Hz pulsing 40 times a second. a short window can't see two frequencies, so it sees their beating in time instead; that's the same signal described from the other side of the tradeoff.

## experiment 2: a chirp past Nyquist

**what people usually predict:** the chirp keeps rising off the top of the plot, or it vanishes above 4 kHz, or "noise".

**what happens:** the chirp rises to 4 kHz at $t = 1$ s, then comes straight back down to 0 at $t = 2$ s, exactly along your `alias_frequency` curve. a 3 kHz tone downsampled from 8 kHz to 4 kHz by keeping every other sample peaks at 1,000 Hz, which `alias_frequency(3000, 4000)` also predicts.

**why:** at true frequency $f \in [4, 8]$ kHz the samples are identical to those of a cosine at $8000 - f$ (on paper 1), so a 6 kHz component is exactly a 2 kHz one as far as the numbers can tell. nothing after sampling can undo that. a resampler must low-pass filter below the new Nyquist frequency *before* it drops samples (that's what `scipy.signal.decimate` and `resample_poly` do).

## experiment 3: pitch or vowel?

**what people usually predict:** the vowel matters more than the pitch in every representation ("/a/ and /i/ are different sounds; /a/ at two pitches is the same sound"), and the DCT in MFCCs "helps" by decorrelating.

**what happens:** (mean distance; ratio per seed in brackets)

| representation | same vowel, two pitches | two vowels, same pitch | ratio (per seed) |
|---|---|---|---|
| log spectrum | 231.5 | 136.5 | 1.70 (1.68, 1.69, 1.72) |
| log mel | 50.9 | 58.5 | 0.87 (0.86, 0.87, 0.87) |
| MFCC c1–c39 | 50.9 | 58.5 | 0.87 (0.86, 0.87, 0.87) |
| MFCC c1–c12 | 29.3 | 57.1 | 0.51 (0.51, 0.52, 0.52) |

in the log spectrum, changing the pitch by 40% moves the vector 1.7 times as far as changing the vowel. the mel filterbank brings the ratio below 1. the DCT changes nothing at all: identical distances to the decimal. truncating to 12 coefficients halves the pitch distance while barely touching the vowel distance.

**why:** a voiced sound's log spectrum is a comb of harmonics (peaks every $f_0$ Hz, valleys 30–50 dB deep between them) riding on the formant envelope. move $f_0$ and the peaks of one comb land in the valleys of the other: hundreds of bins change by tens of dB (the left plot). change the vowel at the same pitch and the combs line up; only the envelope moves, in a few broad regions. the mel filterbank averages several harmonics into each band above the lowest few, which mostly removes the comb (the right plot), though the lowest bands, a few harmonics wide, still see the pitch. the orthonormal DCT is a rotation, and rotations preserve distances (dropping $c_0$ is exactly removing the mean), so on its own it can't change a ratio of distances. the work is done by *discarding* the high coefficients, which hold the fast ripple across bands that's left of the comb.

the loss is welcome for recognizing vowels and words, which shouldn't depend on the speaker's pitch. it's unwelcome for anything that needs the pitch or the fine structure (speaker identification, prosody, music transcription, and resynthesis), which is why speaker recognition systems use more coefficients or other features, and why a codec (topic 26) can't work from MFCCs. one caution: these are synthetic, perfectly periodic vowels with fixed formants. real voices add jitter, breathiness and coarticulation; the direction of the result is the textbook one, but the exact ratios are properties of this toy.
