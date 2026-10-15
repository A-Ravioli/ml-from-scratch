# embeddings · lab notes

numbers are for tiny Shakespeare (the cached file), measured with the reference solution on a laptop CPU: 204,062 words, of which the 3,330 that occur at least 5 times are kept (188,851 tokens, about 1.1 million skip-gram pairs at window 3). training takes about 12 s per model; the whole lab about 40 s. offline, the synthetic stand-in has about 60 words drawn independently of each other, so there is no structure to learn, and none of the analogy words exist (the analogy scores show `nan`).

## experiment 1: what ends up next to what

**what people usually predict:** that "love" is closest to its synonyms ("dear", "affection") and "day" to "morning"; that opposites are far apart.

**what happens** (seed 0):

| word | nearest neighbours (cosine) |
|---|---|
| king | richard (0.70), vi (0.61), edward (0.59), henry (0.57), crown (0.57), sovereign (0.57) |
| love | beauty (0.55), recompense (0.53), hate (0.53), modesty (0.51), sin (0.49), kill (0.49) |
| thou | wilt (0.68), art (0.65), dost (0.64), didst (0.64), camest (0.64), shalt (0.62) |
| night | night's (0.60), untimely (0.57), day (0.57), morning (0.55), morn (0.50), wedding (0.49) |

| pair | cosine |
|---|---|
| love – hate | 0.53 |
| love – dear | 0.37 |
| day – night | 0.57 |
| day – morning | 0.36 |
| good – bad | 0.32 |
| king – queen | 0.41 |
| thou – you | 0.10 |

"hate" is closer to "love" than "dear" is, and "night" closer to "day" than "morning" is. "thou" and "you", the same word in two registers, are nearly unrelated (0.10).

**why:** skip-gram makes two words similar when they predict the same contexts. "love" and "hate" fill the same slots ("I ___ thee", "to ___ him", "my ___ for"), so they get similar vectors; "dear" is mostly an adjective or a form of address and occurs elsewhere. the vectors measure relatedness of usage, and opposites are among the most related words there are. the "thou" result is the same effect from the other side: in Shakespeare's English, "thou" takes its own verb forms ("thou wilt", "thou art", "thou dost"), which "you" never does, so the two have almost disjoint contexts at window 3. the king's neighbours are the play-specific company it keeps: Richard, Henry, Edward, the regnal numbers, the crown.

a detail in the loss curve: the batch loss climbs slowly within each epoch and drops at every epoch boundary. a fixed sample of 20,000 pairs, evaluated during training, falls smoothly instead (2.61 → 2.36 over three epochs). the batch loss is always measured on pairs not yet seen in the current epoch, and the model fits the pairs it has just trained on, mostly those of rare words with few pairs, better than the ones it hasn't: a mild memorization, visible only because the curve is the training loss.

## experiment 2: vector arithmetic

**what people usually predict:** most of the analogies right, as in the famous examples; and that without the exclusion the answer is still usually right.

**what happens** (26 analogies with all four words in the vocabulary):

| model | correct (top 1) | correct in top 5 | nearest without exclusion is a, b or c |
|---|---|---|---|
| seed 0 | 31% | 46% | 92% |
| seed 1 | 31% | 31% | 92% |

the model solves the pronoun analogies (he : him :: she : her, i : my :: we : our, thou : thy :: you : your, do : did :: have : had, boy : girl :: son : daughter) and fails almost all the gender and royalty ones (king : queen :: lord : lady → "elizabeth"; man : woman :: king : queen → "protector") and the irregular verbs (come : came :: go : went → "fetch"). without the exclusion, 24 of the 26 nearest vectors are one of the query words.

**why:** for unit vectors $a$ and $b$ that are close, $b - a$ is short, so $b - a + c$ stays near $c$ (and $b$), and the nearest word is one of the inputs: the exclusion rule is doing much of the work in every analogy result, including the famous ones (Linzen, 2016). what's left is a search in the neighbourhood of $c$ shifted slightly towards $b$. pronouns work because they are frequent and their contexts differ systematically (subject vs object position, possessives before nouns). gender analogies need consistent "male/female" contexts across many word pairs, and 200,000 words of plays don't provide that; the original results used billions of words. with 26 questions each answer is worth 4 points, and the binomial standard error at 31% is about 9 points, so the two seeds' top-5 difference (46% vs 31%, four answers) is within noise.

## experiment 3: prediction or counting?

**what people usually predict:** that the neural model wins clearly, or that the two methods give unrelated vectors.

**what happens:**

| method | analogies (top 1) | analogies (top 5) |
|---|---|---|
| word2vec (seed 0) | 31% | 46% |
| PPMI + SVD (64 dimensions) | 23% | 50% |

the SVD of the 3,330 × 3,330 positive PMI matrix takes about 15 s, as long as training word2vec, and does as well on the analogies (the differences are 2 answers out of 26, well within noise). on average 26% of a frequent word's 10 nearest neighbours are the same under both methods: the two spaces agree on the obvious neighbours ("king": richard, edward, henry, vi in both) and disagree on the long tail. PPMI+SVD neighbours lean towards words that co-occur directly ("love": hate, thy, her, juliet, dear, romeo).

**why:** Levy & Goldberg (2014) showed that skip-gram with $K$ negatives, at its optimum, makes every score $u_c^\top v_w$ equal to $\text{PMI}(w, c) - \log K$ (with the 3/4-smoothed context distribution). word2vec finds a low-rank approximation of that matrix by SGD; the SVD finds the best rank-64 approximation of a closely related one (positive PMI, unshifted) directly. same information, different weighting of errors (SGD weights each cell by how often the pair occurs, the SVD weights all cells equally), so the spaces are related but not identical. what word2vec offers is scale: the SVD needs the whole $|V| \times |V|$ matrix (a million words is $10^{12}$ cells), while SGD streams pairs and only touches the rows it needs.
