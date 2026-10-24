# reading papers

## why this matters

a thesis rests on a few hundred papers you've skimmed, a few dozen you've read properly, and a handful you know as well as their authors do. nobody reads hundreds of papers closely, so the skill isn't reading, it's deciding *how deeply* to read each one, and keeping a record good enough that six months later you can say what a paper claimed, where it showed it, and how much you believed it.

most people never learn this explicitly. they read every paper front to back, highlight half of it, and keep no notes. the costs show up later: citing a paper for a result it never showed (the abstract said "improves", the table said "within noise"), re-reading the same paper three times because nothing stuck, missing the one 2012 paper that everything in your area quietly builds on, and building a reading list from whatever arXiv posted this week.

this topic gives you a method and a small toolkit. the method is Keshav's three passes (2007), plus a structured note per paper that records claims with the *location* of their evidence. the toolkit treats your reading list as data: parse your bibliography, lint your notes, order a reading set so foundations come first, and rank which unread paper your set leans on most. by the end of the chapter you'll have annotated ten papers in your thesis area with it.

## warm-up: predict first

1. how long should a first pass over a 10-page paper take: 5 minutes, 30 minutes, or an hour?
2. a paper's abstract says "our method substantially improves robustness". what's the first thing you look for in the paper to check it?
3. in a reading set of 30 papers on one topic, which paper do you expect to be cited by the most others: one of the newest, one in the middle, or one of the oldest?
4. in a field that grows over time, you rank papers by citation count. of the top 10, how many come from the oldest fifth of the field?

## prerequisites check

- [ ] you can write a loop that scans a string character by character with an index (→ ch 00 · topic 01)
- [ ] you know what a directed acyclic graph is and what a topological order means (→ ch 12 · topic 02)
- [ ] you know a Markov chain's stationary distribution and how power iteration finds it (→ ch 04 · topic 07)
- [ ] you've read at least one paper closely for the R1 milestone, or are about to (→ `milestones/R1.md`)

## core ideas

### idea 1: your reading list is data

every paper you read enters a bibliography, and almost every bibliography in ML is BibTeX: reference managers export it, arXiv and the ACL Anthology serve it, and LaTeX papers cite from it. an entry looks like

```bibtex
@inproceedings{he2016deep,
  title  = {Deep Residual Learning for {Image} Recognition},
  author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  year   = 2016,
  booktitle = "CVPR",
}
```

an `@type`, a citation key, and `name = value` fields, where a value is in braces (which may nest: inner braces protect capitalization from the bibliography style), in double quotes, or bare (a number or a macro like `dec`). a parser turns this into records you can dedupe, link to notes by key, and query. writing it is a nice exercise in careful scanning: braces must match, commas inside braces don't separate fields, and a malformed file must fail loudly instead of looping forever or silently dropping half an entry. duplicate keys are the most common real bug in hand-maintained bibliographies (LaTeX silently uses one of them), so they're an error here.

> **spec.** `parse_bibtex(text)` → a list of dicts, one per entry in file order: `{'type': ..., 'key': ..., field: value, ...}`. every `@` outside an entry starts an entry: `@`, a type (letters only, lowercased), `{`, a body up to the *matching* `}`. `@comment`, `@preamble` and `@string` entries are skipped whole. (so a stray `@` in prose between entries, like an email address, is an error: real BibTeX tools also read it as the start of an entry.) the body is the key (up to the first comma, stripped; non-empty, without whitespace, `=`, braces or quotes) and then comma-separated `name = value` fields: names lowercased; values in `{…}` (braces nest) or `"…"`, or bare (letters, digits, `_-.:`). every value is stored as a string with all `{` `}` removed and runs of whitespace collapsed to one space. text outside entries is ignored. `ValueError` for unbalanced braces, a missing key, a duplicate key, a field that appears twice in one entry (names compared after lowercasing), a field without `=`, an unterminated quote, or a bad type, field name or bare value.

> **▶ build: stage 1.** implement `parse_bibtex`.

### idea 2: three passes, and a note that records evidence

Keshav's method (2007) reads a paper in up to three passes, each deeper and each optional.

- **pass 1, about five to ten minutes.** title, abstract and introduction; section headings; the conclusion; a glance at the figures and the references. the goal is a decision (read on or not) and five facts: what *category* of paper it is (a new method, an analysis, a benchmark, a theory paper, a survey), its *context* (which papers it builds on), whether its assumptions look *correct*, what its main *contributions* claim, and whether it's *clearly* written.
- **pass 2, up to an hour.** read the whole paper but skip proofs and fine details. look hard at every figure and table: are the axes labelled, are there error bars, how many runs, is the comparison fair? mark the references you'll need. you should now be able to summarize the paper's argument, with its evidence, to someone else.
- **pass 3, several hours.** reconstruct the work: re-derive the key equations, imagine implementing the method from the paper alone, and challenge every assumption. this is where you find the unstated details and the gaps between claim and evidence. for the papers your thesis builds on, and for any paper you reproduce, you need this pass.

the record of a reading is a note, and a structured note is more useful than free text because it can be checked and searched. the template here asks for: the citation key, title and year; the category; a summary short enough to force you to compress (at most 120 words by default); the paper's **claims**, each with its **evidence**, which after pass 2 must say *where* the paper supports it ("table 2", "figure 4, left", "appendix B"), because "the experiments show it" is exactly the kind of memory that turns into a mis-citation; the method (pass 2 on); and after pass 3, the assumptions the result depends on, its limitations, and the questions you'd ask the authors. `cites` lists the keys of papers in your reading set that this one cites, which idea 3 turns into a graph.

a linter for this template is a few dozen lines, and it changes how you read: knowing that every claim needs a location makes you look for one.

> **spec.** `check_note(note, *, known_keys=None, max_summary_words=120)` → a list of problem strings, each starting with the offending field and `': '` (empty list = the note passes). rules: `key`, `title`, `category`, `summary` are non-empty strings; `year` is an `int` (not a bool) in 1900–2100; `pass` is 1, 2 or 3 (not a bool; if invalid, report it and check the rest as a pass-1 note); `summary` has at most `max_summary_words` words; `claims` is a non-empty list, and each `claims[i]` is a dict with a non-empty `claim` (problem field `claims[i].claim`). from pass 2 on: every claim's `evidence` matches `LOCATION` (a regex given in the setup cell: a word like figure/fig/table/section/sec/eq/appendix/algorithm/theorem/lemma or `§`, followed by a number or capital letter; field `claims[i].evidence`), and `method` is a non-empty string. at pass 3: `assumptions`, `limitations` and `questions` are non-empty lists of non-empty strings. `cites` (optional, default `[]`) must be a list of strings; if `known_keys` is given, each cited key not in it is a problem naming the key. one problem per broken field. never modify the note.

> **🔮 predict:** you write notes for five papers you read last month, from memory. how many of their claims can you give a location for (a figure, a table, a section) without reopening the paper?

> **▶ build: stage 2.** implement `check_note`.

### idea 3: the citation graph, and reading foundations first

within a reading set, citations form a directed graph: an edge from $a$ to $b$ when paper $a$ cites paper $b$. since papers cite older papers, the graph is almost always **acyclic**, and a topological order (every paper after all the papers it cites) is a reading order in which you never meet a reference you haven't read. there can be many valid orders; making the choice deterministic (alphabetical among the papers that are ready) means the same reading set always gives the same list. cycles do happen in practice (two preprints that cite each other's later versions, or a survey and a paper by the same group), and then there's no valid order for the papers involved: say which ones, and let the reader decide.

**Kahn's algorithm** (1962) builds the order. count, for each paper, how many of its references are still unread. the papers with count zero are ready; repeatedly take the alphabetically first ready paper (a min-heap keeps this cheap), append it to the order, and decrement the count of every paper that cites it, adding those that reach zero to the ready set. if the ready set empties before every paper is placed, the remaining papers each wait for another remaining paper: they contain a cycle.

> **spec.** `citation_graph(notes)`: `notes` is a list of dicts with a `key` and an optional `cites` list. returns a dict (in the notes' order) mapping each key to the **sorted** list of distinct keys it cites that are in the reading set, without self-citations. `ValueError` for duplicate keys. `reading_order(graph)`: a list of all keys in which every key comes after every key it cites, choosing the alphabetically smallest ready key at each step (a duplicated reference counts once). `ValueError` if a cited key isn't in the graph (naming it), or if there's a cycle, with a message naming the keys that couldn't be placed (they include the cycle, and anything that cites into it).

> **▶ build: stage 3.** implement `citation_graph` and `reading_order`.

### idea 4: what to read next

which unread paper does your reading set depend on most? the simplest answer is the in-degree: how many papers in your set cite it. **PageRank** (Page et al., 1999) refines it with a model of a reader who starts at a random paper and, at each step, follows one of its references at random with probability $d$ (the *damping*, usually 0.85) or jumps to a uniformly random paper with probability $1 - d$. the fraction of time the reader spends at each paper is its score. a citation from a paper that cites three others passes on a third of its score; a citation from a survey with 200 references passes on almost nothing. papers that cite nothing in the set (*dangling* papers, often the oldest ones) would trap the reader, so from them the reader jumps uniformly. with $r$ the score vector, $\mathrm{out}(a)$ the number of distinct papers $a$ cites, and $D$ the dangling papers, the scores satisfy

$$r_b = \frac{1 - d}{n} + d \sum_{a \to b} \frac{r_a}{\mathrm{out}(a)} + \frac{d}{n}\sum_{a \in D} r_a,$$

the stationary distribution of a Markov chain (ch 04). power iteration finds it: start from $r = 1/n$, apply the right-hand side until the scores stop changing. Chen et al. (2007) ran PageRank over a century of Physical Review citations and found it surfaces old, foundational papers that citation counts underrate.

one warning before you trust any of these numbers. citations accumulate: an old paper has had decades to collect them, and papers that are already cited get found and cited more (*cumulative advantage*, Price, 1976; with paper-specific "fitness", Bianconi & Barabási, 2001). citation counts across papers of different ages mix quality with age, and the lab measures how much. Newman (2009) suggests comparing a paper only with others published around the same time.

> **spec.** `pagerank(graph, *, damping=0.85, tol=1e-10, max_iters=1000)` → a dict (in the graph's order) of scores summing to 1, computed by power iteration on the equation above from the uniform vector; duplicated references count once; stop when the L1 change of one update is below `tol` or after `max_iters` updates; renormalize to sum 1 at the end. `{}` for an empty graph; `ValueError` for a cited key not in the graph or a damping outside $[0, 1)$. `next_to_read(graph, read, k=5, *, damping=0.85)`: the up-to-`k` keys not in `read`, by decreasing PageRank (computed with `damping`), ties alphabetical. `ValueError` if `read` contains a key that isn't in the graph (a typo would otherwise silently do nothing).

> **🔮 predict:** in a simulated field of 400 papers where each new paper cites 8 older ones, with a preference for good papers and for already-cited papers, how many of the 10 most-cited papers come from the oldest 20%? does PageRank do better?

> **▶ build: stage 4.** implement `pagerank` and `next_to_read`.

## worked example

**a bibliography entry.** `@Article{He2016, Title = {Deep Residual Learning for {Image}\n Recognition}, year = 2016}` parses to `{'type': 'article', 'key': 'He2016', 'title': 'Deep Residual Learning for Image Recognition', 'year': '2016'}`: the type and field name lowercased, the key kept as written, the inner braces dropped, the line break collapsed, the bare year kept as a string.

**a note.** a pass-2 note whose first claim's evidence reads "the results clearly show it" and which has no `method` gets exactly two problems: `claims[0].evidence: …` and `method: …`. change the evidence to "Figure 4, left" and add a method, and it passes. as a pass-1 note it would have passed already.

**a reading order.** resnet cites batchnorm and vgg; vgg cites alexnet; densenet cites resnet and batchnorm. alexnet and batchnorm are ready first: alexnet (alphabetically first), which makes vgg ready; then batchnorm; then vgg, which makes resnet ready; then resnet; then densenet. order: alexnet, batchnorm, vgg, resnet, densenet.

**PageRank by hand.** two papers, $a$ cites $b$, $b$ cites nothing, $d = 0.5$. $b$ is dangling, so $r_a = 0.25 + 0.25\, r_b$ and $r_b = 0.25 + 0.5\, r_a + 0.25\, r_b$. with $r_a + r_b = 1$: $r_a = 0.25 + 0.25(1 - r_a)$, so $r_a = 0.4$ and $r_b = 0.6$. your code must reproduce all four.

## on paper

1. **the dangling fix.** for the two-paper example, write one step of power iteration *without* the dangling term ($b$'s score just disappears). what happens to the total score over many steps, and what does that do to the ranking?

<details><summary>solution</summary>

without the term, $r_a' = 0.25$ and $r_b' = 0.25 + 0.5\, r_a$ (the $1 - d$ part is $(1 - 0.5)/2 = 0.25$ each). starting from $(0.5, 0.5)$: $(0.25, 0.5)$, total 0.75; then $(0.25, 0.375)$, total 0.625, and it settles there: the score that reaches $b$ leaks out every step. but the *ratio* $0.25 : 0.375 = 0.4 : 0.6$ is exactly PageRank's, and that's no coincidence. without the dangling term the fixed point is $x = \tfrac{1-d}{n}(I - dP^\top)^{-1}\mathbf 1$; with it, the dangling mass adds the same amount to every paper, so PageRank is $r = c\,(I - dP^\top)^{-1}\mathbf 1$ for a constant $c$. the leak loses mass, not the ranking: renormalizing once at the end recovers PageRank exactly, on any graph. the real bugs are using the leaky vector *without* renormalizing (scores that don't sum to 1, and that can't be compared across graphs), and renormalizing after *every* step, which converges to a different vector and can change the order (the checks include a five-paper graph where it does).

</details>

2. **every DAG has a reading order.** show that a finite directed acyclic graph always has a node with no outgoing edges (a paper citing nothing in the set), and use it to argue that Kahn's algorithm places every node of a DAG.

<details><summary>solution</summary>

start anywhere and keep following an outgoing edge. in a finite graph you either stop at a node with no outgoing edge or revisit a node, which would be a cycle. so a DAG has a node citing nothing: it's ready. remove it; what's left is still a finite DAG, so it again has a ready node (its count of unread references is zero once the removed nodes are read). by induction the algorithm never runs out of ready nodes until every node is placed. if it does run out early, the remaining graph has no ready node, so by the same argument it contains a cycle.

</details>

3. **your reading budget.** using Keshav's rough times (pass 1: 10 minutes, pass 2: an hour, pass 3: four hours for a paper outside your comfort zone), estimate the time for this topic's deliverable: first passes over 40 candidate papers, second passes over the 10 you annotate, third passes over the 3 your thesis will build on.

<details><summary>solution</summary>

$40 \times 10$ min $= 6.7$ h, $10 \times 1$ h $= 10$ h, $3 \times 4$ h $= 12$ h: about 29 hours, or two to three weeks alongside other work. the point of the passes is the triage: reading all 40 papers closely would take 160+ hours.

</details>

## experiment

the lab turns your toolkit on two citation networks. the first is real: the links to arXiv papers in every lesson of this course, treated as a reading set (lessons cite papers), which gives a census of the papers the curriculum leans on most and shows where PageRank and raw counts disagree. the second is simulated: a growing field in which each paper has a hidden quality and new papers prefer to cite good and already-popular papers. you'll compare citation counts, citations per year, PageRank and an age-adjusted score as ways to find the best papers.

## check your understanding

1. why does the note template ask for the *location* of each claim's evidence rather than a description of it?

<details><summary>answer</summary>

a location can be checked in seconds, by you or a reader; a description can't, and memory drifts towards what the abstract claimed. looking for the location while reading also forces you to notice claims that have none (a sentence in the introduction that no experiment supports).

</details>

2. a paper's first pass leaves you unsure whether its central assumption is right. what do you do next?

<details><summary>answer</summary>

decide whether it matters for your work. if not, record the doubt in the note and move on. if it does, do the second pass with that question in mind (find where the assumption is used and whether any experiment tests it), and the third pass if the paper is central. the passes are a budget, not a ritual.

</details>

3. why might `reading_order` fail on a real reading set, and why is raising an error better than returning some order?

<details><summary>answer</summary>

citation cycles (preprints citing each other's revisions, companion papers). any order would put some paper before one it cites, silently. naming the papers in the cycle lets you decide which to read first, usually the one you can follow without the other.

</details>

4. a survey with 300 references cites every paper in your set. how does it change in-degree rankings and PageRank rankings?

<details><summary>answer</summary>

it adds 1 to every paper's in-degree, so it doesn't change the in-degree ranking, but it does inflate counts. in PageRank it passes only $1/300$ of its score along each reference, so its effect on any one paper is small. PageRank discounts indiscriminate citers; in-degree doesn't.

</details>

5. with damping $d = 0$, what does PageRank return, and what does $d \to 1$ do?

<details><summary>answer</summary>

$d = 0$: the reader never follows a citation, so every paper gets $1/n$. as $d \to 1$ the reader almost never jumps, scores concentrate on the papers that citation paths end at (the oldest foundations and dangling papers), and power iteration converges more slowly (the error shrinks roughly like $d^t$).

</details>

6. you rank papers in your reading set by citation count to pick what to read. what systematic bias do you introduce, and how would you reduce it?

<details><summary>answer</summary>

older papers have had longer to accumulate citations, and cumulative advantage amplifies early leads, so you over-read old papers and miss strong recent ones. compare each paper with papers of a similar age (a z-score within a time window, as in the lab, or citations relative to the median of its year), and for very recent work rely on reading rather than counts.

</details>

## common bugs

- **splitting fields on commas.** `author = {He, Kaiming and ...}` contains commas; only commas at brace depth zero separate fields. scan with a depth counter.
- **a scanning loop without an end condition.** `while depth > 0: ...` on an unclosed brace runs off the end of the string (an `IndexError` if you're lucky, an infinite loop if you index modulo the length or `find` from the same position). check `j < len(text)` in every loop.
- **counting a duplicated reference twice.** if `refs = ['b', 'b']` sets the count to 2 but `b` is only read once, `a` is never ready and you report a false cycle. count distinct references.
- **taking the ready papers in insertion order.** a plain list or queue gives *a* valid order, but not the deterministic alphabetical one; use a heap (`heapq`) or sort.
- **renormalizing at every step.** if dangling papers leak score, renormalizing once at the end gives PageRank exactly (on-paper problem 1); renormalizing after every update converges to a different vector and can reorder papers. the spec spreads the dangling score uniformly at each step (so even a single update, with `max_iters=1`, is pinned down); letting it leak and renormalizing once at the end converges to the same scores, but its intermediate vectors differ.
- **mutating the note.** filling in defaults (`note.setdefault('cites', [])`) changes the caller's data; read with `.get`.

## going deeper

- Keshav (2007), "How to read a paper", *ACM SIGCOMM Computer Communication Review* 37(3): 83–84. the three-pass method, in two pages. [doi:10.1145/1273445.1273458](https://doi.org/10.1145/1273445.1273458)
- Carey, Steiner & Petri (2020), "Ten simple rules for reading a scientific paper", *PLoS Computational Biology* 16(7): e1008032. [doi:10.1371/journal.pcbi.1008032](https://doi.org/10.1371/journal.pcbi.1008032)
- Page, Brin, Motwani & Winograd (1999), "The PageRank citation ranking: bringing order to the web", Stanford InfoLab technical report. [ilpubs.stanford.edu:8090/422](http://ilpubs.stanford.edu:8090/422/)
- Chen, Xie, Maslov & Redner (2007), "Finding scientific gems with Google's PageRank algorithm", *Journal of Informetrics* 1(1): 8–15. [arXiv:physics/0604130](https://arxiv.org/abs/physics/0604130)
- Price (1976), "A general theory of bibliometric and other cumulative advantage processes", *Journal of the American Society for Information Science* 27(5): 292–306. [doi:10.1002/asi.4630270505](https://doi.org/10.1002/asi.4630270505)
- Bianconi & Barabási (2001), "Competition and multiscaling in evolving networks", *Europhysics Letters* 54(4): 436. the fitness model the lab simulates. [arXiv:cond-mat/0011029](https://arxiv.org/abs/cond-mat/0011029)
- Newman (2009), "The first-mover advantage in scientific publication", *Europhysics Letters* 86: 68001. comparing papers with their contemporaries. [arXiv:0809.0522](https://arxiv.org/abs/0809.0522)
- Kahn (1962), "Topological sorting of large networks", *Communications of the ACM* 5(11): 558–562. [doi:10.1145/368996.369025](https://doi.org/10.1145/368996.369025)

## connections

- ← builds on: ch 04 · topic 07 (Markov chains and stationary distributions), ch 12 · topic 02 (directed graphs), and every paper you've read for R1 and R3.
- → used in: topic 02 (the first step of the reproduction protocol is a three-pass reading), topic 04 (a research question starts from a gap you found while reading), topic 05 (reviewing is pass 3 written down), and the thesis capstone's related-work section, built from your notes.
