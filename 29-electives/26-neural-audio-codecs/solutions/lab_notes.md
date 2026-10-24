# neural audio codecs · lab notes

numbers are from the reference solution, one CPU thread; the lab takes 60–110 s depending on machine load, most of it training two codecs. **everything is synthetic**: 128 ms clips of stationary harmonic vowels, chirps and resonant noise, each at a fixed loudness. that is a far easier and more regular signal than speech or music, and it favours some codecs over others (see experiment 2). no perceptual metric (PESQ, ViSQOL, listening tests) was run; the lesson's idea 3 says why that matters.

## experiment 1: residual vs single codebooks

**what people usually predict:** RVQ is much worse than a single codebook at the same bits ("it can only reach sums"), and every stage adds the same number of dB, like the 6 dB per bit of scalar quantization.

**what happens:** (SNR in dB on 8-sample waveform frames, training / held-out test)

| quantizer | bits per frame: train / test SNR (dB) |
|---|---|
| RVQ, 16 codes per stage | 4: 4.3 / 4.0, 8: 7.3 / 6.9, 12: 9.9 / 9.3, 16: 12.2 / 11.4, 20: 14.3 / 13.3, 24: 16.2 / 15.0 |
| RVQ, 4 codes per stage | 2: 1.8 / 1.6, 4: 3.1 / 2.8, 6: 4.3 / 4.0, 8: 5.4 / 5.1, 10: 6.5 / 6.0, 12: 7.6 / 7.0, 14: 8.6 / 8.1, 16: 9.5 / 8.9 |
| RVQ, 256 codes per stage | 8: 9.6 / 8.3, 16: 15.8 / 13.2 |
| one codebook | 4: 4.3 / 4.0, 6: 7.1 / 6.5, 8: 9.6 / 8.3, 10: 13.3 / 10.1 |

at 8 bits, 2 × 16 RVQ is 1.4 dB behind one 256-codeword codebook on test frames (6.9 vs 8.3), using 32 codewords instead of 256; 4 × 4 RVQ is 3.2 dB behind with 16 codewords. each 4-bit stage adds less than the previous one: 4.0, 2.9, 2.4, 2.1, 1.9, 1.7 dB on test frames. and the single codebook's train–test gap explodes as it grows: 1.3 dB at 8 bits, 3.2 dB at 10 bits, against 1.2 dB for the 16-code RVQ even at 24 bits.

**why:** the first stage captures the large, structured variation (the overall shape and level of a frame); later stages see residuals that are smaller and closer to unstructured noise, for which a 16-codeword codebook has little to exploit, so the dB per stage shrink. the per-bit rate is also far below scalar quantization's 6 dB per bit because these bits are per *frame* of 8 samples: one extra bit per sample would be 8 extra bits per frame. the RVQ penalty at equal bits is modest because the reachable sums still tile the data's high-density regions well, and greedy encoding usually finds a good tuple. the overfitting is about samples per parameter: at 10 bits, the single codebook has 1,024 codewords for 8,000 training frames, about 8 frames each, so many codewords memorize a handful of frames; the 16-code RVQ gives each codeword 500 frames per stage. extrapolate to 80 bits per frame and the single codebook needs $2^{80}$ codewords (and data to fit them), while RVQ needs 20 stages of 16.

## experiment 2 (project): bitrate against quality

**what people usually predict:** the neural codec wins everywhere ("it's learned"), or at least at low bitrates on both metrics; μ-law improves steadily with bits.

**what happens:** (200 test clips; ConvCodec rows average two training seeds, per-seed SNR / LSD in brackets)

| codec | kbps | SNR (dB) ± s.e. over clips | LSD (dB) |
|---|---|---|---|
| PCM | 8 | −12.9 ± 0.2 | 31.5 |
| PCM | 16 | −5.4 ± 0.2 | 26.0 |
| PCM | 24 | 2.4 ± 0.2 | 21.0 |
| PCM | 32 | 8.3 ± 0.1 | 17.5 |
| μ-law | 8 | 4.2 ± 0.1 | 16.6 |
| μ-law | 16 | −0.5 ± 0.2 | 21.4 |
| μ-law | 24 | 8.4 ± 0.1 | 16.2 |
| μ-law | 32 | 13.7 ± 0.1 | 13.6 |
| k-means RVQ, raw frames | 2 | 4.5 ± 0.2 | 14.4 |
| k-means RVQ, raw frames | 4 | 7.3 ± 0.2 | 13.6 |
| k-means RVQ, raw frames | 8 | 11.0 ± 0.3 | 12.2 |
| k-means RVQ, raw frames | 16 | 16.3 ± 0.4 | 10.4 |
| your ConvCodec | 2 | 2.6 ± 0.2 | 11.5 (2.7, 2.6 / 12.1, 10.8) |
| your ConvCodec | 4 | 4.5 ± 0.3 | 11.2 (4.5, 4.5 / 12.0, 10.4) |
| your ConvCodec | 8 | 6.5 ± 0.4 | 11.0 (6.3, 6.6 / 11.9, 10.1) |
| your ConvCodec | 16 | 7.4 ± 0.5 | 10.8 (7.3, 7.6 / 11.7, 9.9) |
| ConvCodec, quantizer bypassed | ∞ | 7.7 ± 0.5 | 10.6 (7.5, 7.9 / 11.5, 9.7) |

the two metrics disagree. on SNR, the codec with no neural network at all, k-means RVQ on raw 32-sample frames, wins at every bitrate, by 1.9 dB at 2 kbps and 8.9 dB at 16 kbps. on log-spectral distance (with the lab's floor eps = 1e-4), your ConvCodec wins at 2, 4 and 8 kbps (11.5 against 14.4 dB at 2 kbps, a gap larger than its seed spread) and ties at 16. at 2 kbps it has a lower LSD than μ-law at 32 kbps, sixteen times the rate, while its SNR is below 1-bit μ-law's. its curve is nearly flat: going from 2 to 16 kbps gains 4.8 dB of SNR and 0.7 dB of LSD, and with the quantizer bypassed it reaches only 7.7 dB SNR. finally, μ-law gets *worse* from 1 to 2 bits per sample.

**why:** start with the ceiling. the "quantizer bypassed" row is the trained autoencoder with unlimited bits, and it's no better than 7.7 dB SNR: after 1,400 Adam steps (about 25 seconds per seed) the encoder and decoder simply can't reproduce these waveforms precisely, and no number of RVQ stages can recover what the decoder can't produce. SoundStream and EnCodec train for hundreds of thousands of steps on GPUs, with larger networks and adversarial losses; this one is a toy, and that's the main reason it loses on SNR. k-means on raw frames has no such ceiling: every extra stage keeps paying (16.3 dB at 8 stages, still rising), and these synthetic clips are stationary and periodic, so 256 codewords per stage cover their frames unusually well.

the LSD win is the codec's training objective showing through. the multi-scale spectral loss rewards matching magnitudes at several resolutions and ignores phase within a frame; the waveform L1 term is only part of the loss. the network learns to produce the right spectral envelope and harmonics even when its waveform is off, which is what LSD measures and SNR punishes (on paper 4). k-means on raw frames minimizes waveform error frame by frame, with blocking at the frame edges and no notion of a spectrum, so at low rates its errors are broadband and its LSD is poor. the LSD ranking also depends on the floor `eps` (the lab's dropdown recomputes it):

| eps | k-means RVQ 2 / 4 / 8 / 16 kbps | your ConvCodec 2 / 4 / 8 / 16 kbps | μ-law 32 kbps |
|---|---|---|---|
| 1e-2 | 6.4 / 5.6 / 4.4 / 2.9 | 6.7 / 6.2 / 5.8 / 5.5 | 4.2 |
| 1e-4 (used above) | 14.4 / 13.6 / 12.2 / 10.4 | 11.5 / 11.2 / 11.0 / 10.8 | 13.6 |
| 1e-6 | 21.8 / 21.0 / 19.6 / 18.0 | 16.1 / 16.1 / 16.3 / 16.2 | 21.6 |

with eps = 1e-2 only bins within about 35 dB of the loudest count, and k-means wins at every rate; the lower the floor, the more the quiet bins between harmonics count, and the larger the ConvCodec's lead. so its advantage lives in the low-level spectral detail, the part its log-magnitude loss term rewards, not in the loud partials. which of the two a listener would prefer at 2 kbps can't be read from these numbers: you'd need a listening test (or at least a perceptual model like ViSQOL), and real speech rather than synthetic tones. that's the honest summary of the project: *on synthetic audio, with 2 seeds × ~25 s of CPU training, a learned codec wins the spectral metric at low rates and loses the waveform metric to k-means on raw frames at every rate (it does beat μ-law on SNR at 8 and 16 kbps); which matters is a perceptual question this lab can't settle.*

μ-law's dip is a property of midrise quantization with very few levels. at 1 bit the two levels are ±0.059, so every sample becomes its sign scaled to roughly the signal's typical magnitude (RMS 0.1): crude, but the right size. at 2 bits the μ-law levels are ±0.012 and ±0.247, and most samples are rounded either far too small or far too large. with 3 and 4 bits there are enough levels near the signal's range for the companding to pay off. plain PCM at 1–2 bits is worse still, because its levels (±0.5 at 1 bit) are five times the signal's RMS.

## experiment 3: how many bits do the tokens really need?

**what people usually predict:** close to 8 bits for every stage (the codebooks are fully used, so the codes look random), and not much to gain from context.

**what happens:**

| codec | stage | entropy (bits) | perplexity | fraction used | bigram cross-entropy (bits) |
|---|---|---|---|---|---|
| your ConvCodec | 1 | 7.28 | 155 | 1.00 | 5.75 |
| your ConvCodec | 2 | 7.34 | 162 | 1.00 | – |
| your ConvCodec | 4 | 7.38 | 167 | 1.00 | – |
| your ConvCodec | 8 | 7.67 | 204 | 1.00 | – |
| k-means RVQ, raw frames | 1 | 7.57 | 190 | 0.95 | 6.07 |
| k-means RVQ, raw frames | 2 | 7.35 | 164 | 0.89 | – |
| k-means RVQ, raw frames | 4 | 7.01 | 129 | 0.86 | – |
| k-means RVQ, raw frames | 8 | 6.99 | 127 | 0.96 | – |

(stages 3, 5, 6 and 7 are left out; for your codec they fall between their neighbours, for k-means they stay between 6.95 and 7.12 bits.) every stage's marginal entropy is 7.0–7.7 bits: an entropy coder that treats codes as independent saves only 4–12%. but the first stage's codes are far more predictable from the previous frame: a bigram model needs 5.75 bits for your codec's first stage and 6.07 for k-means's, about 20% less than the marginal entropy.

**why:** a 128 ms synthetic vowel or chirp changes slowly, so consecutive frames tend to land on the same or related codewords, and the code sequence has strong temporal structure even though every codeword is used about equally often overall. that structure is exactly what an entropy model over the codes (EnCodec's small transformer) or a codec language model exploits: AudioLM and VALL-E predict codes from their context, and a model that predicts well is a model that compresses well. for your codec the finer stages have the highest entropy: they encode residuals that look more like noise, and dead-code restarts keep all 256 codes in use, so their marginals stay flat. k-means shows the opposite trend (7.57 bits at stage 1 down to about 7.0 from stage 4 on), with its codeword usage growing more uneven (perplexity 190 at stage 1, about 125–130 from stage 4 on; nothing restarts its rarely used codewords), so "residuals look like noise" is a tendency of this codec, not a law. either way, codec language models spend most of their modelling effort on the coarse stages, where the structure is. a bigram over one stage is the weakest possible model: a transformer conditioning on many past frames and on the coarser stages of the same frame would do much better, and real speech, with its phoneme and word structure, has far more context to exploit than these random clips.
