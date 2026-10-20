# probability & random variables · lab notes

## experiment 1: the base-rate fallacy

**what people usually predict:** something close to the test's sensitivity, 80–90%, at every prevalence. people who know the trick often guess "around 50%".

**what happens:** at 1% prevalence, $P(\text{sick} \mid +) = 0.0917$. the simulated population agrees to two decimals. at 0.1% it's 0.0099, so 99 of every 100 positives are false alarms. the curve crosses 0.5 only at a prevalence of $1/11 \approx 9.1\%$, and even at 50% prevalence it's 0.91, not 0.9.

**why:** per 1000 people at 1% prevalence, 10 are sick and 9 of them test positive. 990 are healthy and 9% of them, about 89, test positive too. so 9 of about 98 positives are real. in odds form: the likelihood ratio is $0.9/0.09 = 10$, so a positive test multiplies the prior odds by 10. the posterior odds reach $1:1$ when the prior odds are $1:10$, a prevalence of $1/11$. the test's "90%" is $P(+ \mid \text{sick})$, a property of the test. $P(\text{sick} \mid +)$ is a property of the test *and the population*. this is why screening programs for rare conditions confirm every positive with a second, more specific test, and why a classifier's precision on a rare class can be terrible even when its recall and false-positive rate look excellent (ch 07).

## experiment 2: Monty Hall, two hosts

**what people usually predict:** for the informed host, 1/2 ("two doors left, so it's 50/50"). people who know the classic answer often predict 2/3 for *both* hosts, since they see the same thing.

**what happens:** informed host: 0.6667 exactly, 0.668 simulated. ignorant host: 0.5000 exactly, 0.505 simulated. and the simulation keeps far fewer of the ignorant host's games (about 16,600 of 50,000, against 25,000).

**why:** what counts as evidence is the probability of the observation under each hypothesis. informed host, observation "opened door 2": likelihoods $1/2, 1, 0$ for the car behind doors $0, 1, 2$. if the car is behind 1, he was *forced* to open 2, so door 1 gets twice the weight of door 0. ignorant host, observation "opened door 2, goat": likelihoods $1/2, 1/2, 0$. his choice ignored the car, so the only information is "the car isn't behind 2", and doors 0 and 1 stay symmetric. the ignorant host reveals the car a third of the time, and those games don't match the observation, which is why fewer are kept. the general lesson: **the same data can mean different things depending on how it was collected.** in ML terms, the data-generating process (how examples were selected, labeled, or filtered) is part of the model. ignoring it is how selection bias gets in.

## experiment 3: adding up dice

**what people usually predict:** something like 20–30 dice before the bell shape appears. for the 10-dice question, guesses range widely, often 15–30%, because "it's the most likely value".

**what happens:** two dice already give a triangle, and at three dice the exact pmf is within 0.008 of the bell curve everywhere (the peak is 0.125). by 10 dice the largest gap is about 0.001. yet the single most likely total of 10 dice, 35, has probability only 0.0727.

**why:** each addition is a convolution, and convolution smooths: it averages shifted copies of the distribution, filling in gaps and rounding corners. a flat, symmetric, bounded distribution like a die has nothing to fight that smoothing (no skew, no heavy tail), so the bell appears after just a few terms. this is the central limit theorem (topic 05) at work, and topic 05's lab shows distributions where it takes much longer, or never happens. as for the peak: the total of 10 dice spreads over 51 values with standard deviation $\sqrt{10 \cdot 35/12} \approx 5.4$, so even the mode only gets about $1/(\sqrt{2\pi} \cdot 5.4) \approx 0.074$. "most likely" and "likely" are very different things once there are many outcomes. the same fact explains why a language model's single most likely sentence usually has a tiny probability (and is often dull), which comes back in the capstone.
