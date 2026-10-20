# speech recognition with CTC · lab notes

numbers are from the reference solution, one CPU thread; the lab takes 30–90 s depending on machine load, almost all of it training four small models (a one-layer bidirectional GRU with 64 units per direction, 16 epochs over 400 synthetic utterances). the corpus is fixed (seeds 0 and 1); the two model seeds change the initialization and the batch order. **everything is synthetic**: eight "phonemes" that are far easier to tell apart than real ones, clean gaps between them, no coarticulation. the error rates below are properties of this toy and say nothing about real speech; the *mechanisms* (what the blank does, peaky posteriors, greedy ≈ beam) are the ones reported for real recognizers.

## experiment 1: what is the blank for?

**what people usually predict:** without a blank the model fails on repeated phonemes (that much follows from the collapse rule), and does about as well as the normal model everywhere else, since "there's no repeat to separate".

**what happens:** (greedy decoding; per-seed values in brackets)

| model | PER, all | PER, utterances with repeats | PER, without repeats |
|---|---|---|---|
| with blank | 0.055 (0.044, 0.065) | 0.066 (0.049, 0.082) | 0.029 (0.034, 0.025) |
| without blank | 0.343 (0.356, 0.329) | 0.434 (0.438, 0.431) | 0.130 (0.167, 0.093) |

without blank, 66% of the utterances with repeats come out shorter than their reference. and the blank-free model is also 4–5 times worse on utterances that contain no repeats at all.

**why:** the first result is structural. a run of a-frames collapses to a single "a", so the only way to say "a a" is to put a blank frame between them, and here a blank frame costs 30 nats. the model learns to give up on the second a: most repeat utterances lose a phoneme, and those deletions are most of the 0.43.

the second result is the more instructive one. without a blank, *every* frame must be labeled with some phoneme: the silence at both ends, the gaps between phonemes, and the transitions. the model has to decide which neighbouring phoneme each of those frames belongs to, and a single stray frame of the wrong class anywhere becomes an insertion or a substitution after collapsing. with a blank, all of those frames have a cheap, harmless label that means "nothing new". so the blank does two jobs: it separates repeats, and it absorbs every frame that isn't evidence for a label. the second job turns out to matter even when there are no repeats. (the blank-free variant here is "CTC with the blank priced out", not a carefully designed alternative; HMM systems solve the same problems with silence models and multiple states per phoneme.)

## experiment 2: what does the trained model output?

**what people usually predict:** the label's probability is high across the phoneme's whole span (about 70% of frames), with blanks in the gaps.

**what happens:**

| seed | frames labeled blank (argmax) | frames inside a phoneme | non-blank frames per phoneme |
|---|---|---|---|
| 0 | 0.893 | 0.692 | 1.24 |
| 1 | 0.898 | 0.692 | 1.18 |

the model labels about 90% of all frames blank, including most frames in the middle of a phoneme. each phoneme gets a single sharp spike, about one frame wide (1.2 non-blank frames per reference phoneme, including the occasional error), and in the plot the spikes sit near the *end* of each phoneme's span.

**why:** this is CTC's well-known **peaky behaviour** (Zeyer, Schlüter & Ney, 2021). the loss sums over alignments, and on paper 3 shows the gradient pushes each $y_t$ toward the posterior over alignments, $\gamma_t$. early in training, when the outputs are nearly uniform, every alignment is equally likely, and at any given frame the blank is the one class that many alignments agree on: any alignment may use it anywhere, while each label is plausible only in the stretch of frames where its position in the sequence can fall, and the label mass is split eight ways. so $\gamma_t$ puts the most mass on blank at most frames, the update makes blank more likely everywhere, which makes blank-heavy alignments more probable still, and so on. nothing in the loss rewards spreading a label over its whole span: one frame of "a" surrounded by blanks scores exactly as well as eight frames of "a". the spike's position is where the bidirectional model has gathered enough evidence to commit, which for this model is late in the segment; it is not a reliable phoneme boundary. that's why CTC models make poor forced aligners without extra constraints, and why "when did the model emit a" can differ from "when was a said" by tens of milliseconds.

## experiment 3: greedy or beam?

**what people usually predict:** beam search helps a few points ("it searches more").

**what happens:**

| seed | PER, greedy | PER, prefix beam (8) | utterances where they differ |
|---|---|---|---|
| 0 | 0.0443 | 0.0443 | 0 of 150 |
| 1 | 0.0650 | 0.0635 | 1 of 150 |

beam search changes one output in 300 utterances.

**why:** the worked example's failure needs probability spread over many paths of one labeling, with no single path dominant. experiment 2 shows the opposite: posteriors near 0 or 1 at almost every frame, so one path carries nearly all of its labeling's probability, and the best path and the best labeling coincide. a beam pays off when it can add information the acoustic model lacks. in practice that's a language model or a lexicon scored as prefixes grow (Hannun et al., 2014), and it pays off on real speech, where the acoustic posteriors are far less certain than on this toy. on this corpus the phoneme sequences are random, so there is nothing for a language model to learn either.
