# BERT-style encoders · lab notes

numbers below are from the reference solution on a heavily shared laptop CPU, on the real SMS spam collection, seed 0, with messages truncated to 63 characters. one seed per cell: treat differences below about 0.03 as noise. an earlier run with 95-character messages (and 800 pretraining steps, 150 fine-tuning steps) gave the same pattern with higher numbers across the board: 0.836 / 0.923 / 0.832 at 40 labels for scratch / pretrained / logistic regression, 0.876 / 0.915 / 0.902 with all labels.

## experiment 1: what masked-language-model pretraining buys

**what people predict.** "pretraining helps a lot with few labels and not at all with many", and "a transformer beats a logistic regression". the first is roughly right; the second isn't.

**what happens.** the MLM loss falls from 3.74 to 2.98 nats in 600 steps ($\ln V = 4.78$). balanced accuracy on the 1,114 test messages (0.5 = guessing):

| labelled messages | from scratch | pretrained | char-count logistic regression |
|---|---|---|---|
| 40 | 0.781 | **0.843** | 0.749 |
| 160 | 0.753 | **0.858** | 0.802 |
| 640 | 0.849 | **0.875** | 0.821 |
| 4,460 | 0.856 | **0.879** | 0.855 |

the pretrained encoder is best at every budget, by 6–10 points with 40–160 labels and by 2–3 points with 640 or more. from scratch, the transformer is barely better than a logistic regression on character counts, and worse than it at 160 labels; with all the labels the two are tied. the pretrained model with 40 labels is about as good as the scratch model with all 4,460.

**why.** pretraining never saw a label, but to fill in characters the encoder has to learn which characters go together: that digits come in runs (phone numbers, prices, "08712"), that capitals cluster, how common words are spelled. those are exactly the surface features that separate spam ("FREE entry", "call 09061...", "£1000") from conversation, so the fine-tuning head starts from representations where the classes are already partly separated, and 40 examples are enough to find the direction. from scratch, 40 examples must also train the embeddings and attention: too many parameters for the data (high variance, on paper 3). as labels grow, the data alone can teach those features and the gap closes. the logistic regression is the reality check: spam detection on SMS is mostly a surface-statistics problem, so the counts of each character get most of the way, and a transformer trained from scratch on this little data doesn't add much. a paper reporting the transformer's 0.86 against a majority-class baseline (0.5 balanced accuracy) would look impressive and mean little; reported against the logistic regression, the honest headline is "pretraining is what helps".

## experiment 2: what the pretrained encoder predicts

**what people predict.** something like 40–60% of hidden characters recovered: English is redundant, and the model sees both sides.

**what happens.** 18.6% of hidden characters are recovered exactly, against 18.2% for always guessing a space, the most frequent character. the filled-in characters are mostly spaces and a few common letters. the encoder that improved fine-tuning by up to 10 points is, at fill-in-the-blank, barely better than a frequency table.

**why.** 600 steps of MLM on 4,460 short messages is very little pretraining: the loss curve shows the model has learned the character frequencies (the unigram entropy of these messages is 3.30 nats, and most of the drop from 4.78 is just that) and only a little about the neighbouring characters (another 0.3 nats, mostly confidence about which letter class fits, not enough to change many top-1 guesses). in a side run with 3,000 steps (learning rate $10^{-3}$, batches of 32), the loss sat near 2.9 for about 2,000 steps and then fell to 2.53 as the model learned to attend to its neighbours: a plateau-then-drop that's common in transformer training, too slow for a CPU lab. so why did this barely-trained encoder help classification? it has embeddings in which characters are organized by frequency and co-occurrence (digits near digits, capitals near capitals), layer norms and attention in a well-conditioned state, and an output that already encodes the character statistics of each message, which for spam is most of the signal (experiment 1's logistic regression). that is a weaker kind of transfer than "it understands language", and an honest reading of the experiment says so. masked prediction is still the right objective for a bidirectional model: with next-character prediction it could simply attend to the next position and copy it (lesson, on paper 1), while a masked character is absent from the input, so the only way to predict it is to model the context.
