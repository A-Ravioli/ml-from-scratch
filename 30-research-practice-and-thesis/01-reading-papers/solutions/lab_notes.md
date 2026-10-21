# reading papers · lab notes

numbers below are from the reference solution. experiment 1 reads the lessons in the repository, so its numbers depend on which chapters exist and will drift as chapters are added: treat its counts as approximate, measured at the time of writing (about 205 lessons). the qualitative pattern is what matters. experiment 2 is a simulation with fixed seeds and is exactly reproducible. the lab runs in about a second.

## experiment 1: what does this course read?

**what people predict.** "Attention is all you need", linked from many lessons; and that PageRank agrees with the count.

**what happens.** at the time of writing, about 205 lessons link to about 500 distinct arXiv papers, nearly 90% of them from a single lesson. the most-linked paper is Baydin et al.'s survey "Automatic differentiation in machine learning" (5 lessons), then three papers with 4: Zhang et al. (2017) on rethinking generalization, Goyal et al. (2017) on large-minibatch SGD, and Vaswani et al. (2017). PageRank agrees on the top two but drops "Attention is all you need" and the large-minibatch paper out of its top 10, and promotes half a dozen papers linked from only 2 or 3 lessons, such as Buitinck et al.'s "API design for machine learning software" (linked from ch 07 · 01 and 07 · 05) and Snoek et al.'s "Practical Bayesian optimization" (07 · 06 and 09 · 06).

the reading queue after chapters 00–18 (roughly a third of the papers already met) suggests papers cited by a single later lesson, for instance Newman's "first-mover advantage" paper from this very topic.

**why.** most of the course's links are one-offs: each lesson points at the papers specific to its topic, and only a handful of papers are cross-cutting. PageRank splits a lesson's "vote" over everything it links to: a lesson with 4 arXiv links passes a quarter of its score to each, one with 20 links a twentieth. the promoted papers are linked from lessons that link to few papers (ch 07's lessons cite mostly books and journals), so each of their links weighs more. whether that's sensible depends on the question: as a measure of "what is this lesson built on" it is (a lesson citing 4 papers depends on each of them more), as a measure of "what does the course rely on most" the plain count is the better reading-queue signal. the queue's oddities have the same cause, plus a second one: the lessons themselves are nodes, and the jump term spreads score evenly over 205 lessons and 505 papers, so a paper linked from one sparse lesson can outrank one linked from three dense ones. notice the census's blind spot too: classic papers cited through a DOI, JMLR or a proceedings page (dropout's JMLR paper, most statistics papers) don't appear at all, which is why "most-linked arXiv paper" isn't "most important paper".

## experiment 2: do citation counts find the best papers?

**what people predict.** that a few of the top 10 come from the oldest fifth (most guess 3–5), and that PageRank or citations per year correct the age effect.

**what happens.** averaged over 5 simulated fields at γ = 1:

| ranking | Spearman ρ with quality | ρ with age | top 10 from the oldest 20% | top 10 that are truly top-10 quality |
|---|---|---|---|---|
| citations | 0.35 | +0.59 | 10.0 | 1.6 |
| citations per year | 0.36 | +0.54 | 9.8 | 1.8 |
| PageRank | 0.34 | +0.60 | 10.0 | 1.2 |
| age-adjusted z | 0.40 | −0.11 | 2.0 | 4.2 |

**all ten** of the most-cited papers come from the oldest fifth of the field in every one of the five simulations, though by chance only about 2 of the 10 best papers do. citations per year barely help, and PageRank is slightly *worse* than the raw count. comparing each paper with the 40 papers published around it removes the age correlation and finds 4.2 of the 10 best papers instead of 1.6.

varying γ (5 fields each; run separately with the lab's `simulate_field`): with no preference for already-cited papers (γ = 0), raw counts still put 8.8 of the top 10 in the oldest fifth and PageRank 10; citations per year improve (ρ = 0.63) and the age-adjusted score reaches ρ = 0.72 and finds 6.6 of the 10 best. at γ = 1.5 nothing works: every ranking has ρ ≈ 0.2 with quality, and even the age-adjusted score finds 1.4 of the 10 best.

**why.** two effects compound. *exposure*: paper 0 can be cited by all 399 later papers, paper 399 by none, so counts grow with age even when every paper is equally good (that's the γ = 0 bias). *cumulative advantage*: with γ > 0, every citation makes the next more likely, so early papers that got lucky snowball, and their lead grows faster than linearly in time; dividing by age removes a linear trend, not a snowball. PageRank makes it worse because credit flows *down* citation chains: a recent paper citing a mid-aged one which cites an old one passes score to all of them, and the oldest papers sit at the end of every chain. the age-adjusted score works because it only asks "is this paper cited more than its contemporaries?", which both effects leave mostly intact (Newman, 2009). and at γ = 1.5 early luck dominates quality so thoroughly that no ranking from citations can recover it: the information isn't in the counts any more.

the practical lesson for building a reading list: raw counts and PageRank tell you what's *foundational*, which is worth knowing (you should read those), but they systematically under-rank strong recent work. for recent papers, compare with papers from the same year, follow what the best recent papers you've read cite, and read, rather than count.
