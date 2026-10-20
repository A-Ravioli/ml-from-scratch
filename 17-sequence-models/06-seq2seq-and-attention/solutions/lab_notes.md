# seq2seq & attention · lab notes

numbers are measured with the reference solution on a laptop CPU; the lab uses only synthetic data, so it gives the same results offline. it takes a little over a minute on an idle machine (about 55 s of training, 13 s of beam search). the 600 test dates are drawn so that none of their source strings occurs verbatim in the training set; about 9.5% of randomly drawn test strings would otherwise be exact repeats.

## experiment 1: with and without attention

**what people usually predict:** that both models learn the task, attention a bit faster; or that without attention the model learns the fields in source order (the day or month first, since they're read first).

**what happens** (greedy decoding on all 600 test dates):

| model | year | month | day | whole date |
|---|---|---|---|---|
| with attention (400 steps) | 1.00 | 1.00 | 1.00 | **1.00** |
| without attention (1,500 steps) | 1.00 | 0.26 | 0.07 | **0.01** |

with attention, the whole date is right for 85% of the test dates after 100 steps and for essentially all of them from step 200 on: per format, 1.00 in five of the six and 0.99 for "the 3rd of march 1987" (one date wrong out of 87). without attention, the year is perfect from the first evaluation (step 100), the month and the day sit near chance for over a thousand steps, and after 1,500 steps (almost four times the attention model's budget) only 1% of dates are fully right. trained on, it does get there: in a separate run of the same model, training data and seed (scored on 300 test dates from an earlier draw that wasn't deduplicated against training), exact match was 0.18 at 2,000 steps, 0.68 at 2,500, 0.98 at 3,000 and 1.00 at 4,000, with the month learned before the day. so attention isn't needed for *capacity* on dates this short; it makes learning roughly ten times faster.

**why:** without attention, the decoder knows only the encoder's final state. in all six formats the year is the last thing the encoder reads, so it's present in the final state with almost no decay, and the gradient from the year's output digits reaches the year's input characters through a handful of steps. the month and the day were read 5–25 steps earlier; for them to survive into the final state, the encoder has to learn to carry them through all the later updates, and the only teacher is a gradient that has to travel back just as far (topics 04–05). that's why the year, read last, is learned first, and the fields read earlier come much later (of those two, the month is learned before the day, in this run; the experiment doesn't isolate why, so "reverse reading order" is only exact for the year against the rest). with attention, every decoder step has a one-step path to every encoder state, so no information has to be carried, and every field is as easy as the year.

## experiment 2: what the attention looks at

**what people usually predict:** either that attention is diffuse and uninterpretable, or that the month digits attend to the first letters of the month name.

**what happens:** the maps are sharp and sensible. while writing the year digits, the attention sits on the year characters of the source, one position to the left of the digit being written (the query is the previous decoder state, so the model looks at "where it is" when deciding the next digit). for the numeric formats ("10/04/1956") the month and day digits attend to the corresponding source digits. for spelled-out months ("saturday, january 8, 2005", "the 16th of june 2043") the month digits attend to the *end* of the month name and the following space or comma, not its first letters, and the day digits to the day number and the characters after it. the output separators (`-`) and the final `eos` get diffuse attention.

**why:** nothing tells the model which source characters correspond to which output characters, but attending to the right place is simply the easiest way to reduce the loss, so gradient descent finds the alignment. the encoder is a left-to-right GRU: its state at the last letter of "january" is the first state that has seen the whole word, so that's where "this is month 01" is readable. the attention map shows where the *information* is in the encoder states, not which characters a human would point to. separators and `eos` are the same for every date, so the decoder doesn't need the context to produce them, and nothing pushes their attention to be sharp.

## experiment 3: greedy decoding against beam search

**what people usually predict:** a few points of improvement from beam search for both models, more for the weaker one.

**what happens** (exact match on 120 test dates):

| model | beam 1 (greedy) | beam 3 | beam 10 |
|---|---|---|---|
| with attention | 1.000 | 1.000 | 1.000 |
| without attention (1,500 steps) | 0.000 | 0.025 | 0.025 |

nothing to gain for the attention model, and almost nothing for the under-trained one.

**why:** beam search finds outputs with higher model probability; it can't add knowledge the model doesn't have. the attention model already puts nearly all its probability on the right date, so every beam agrees with greedy decoding. the model without attention doesn't know the day: its distribution over day digits is spread out, and the most probable complete date under that distribution is still wrong, whatever the search. beam search pays off when the model has the right answer somewhere near the top but greedy commits to a locally attractive first token, as in translation, where an early word choice constrains the whole sentence (Sutskever et al., 2014, report clear gains from small beams). on open-ended language-model text (chapter 27) it has the opposite problem: it finds high-probability text that is bland and repetitive, because the most probable continuation is rarely the most natural one (Holtzman et al., 2020).
