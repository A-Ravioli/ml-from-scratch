# reading papers · hints

one hint per stage. at this level the spec in the lesson is the main help; open a hint only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

think in two levels. first find each whole entry: from an `@`, read the type up to `{`, then walk forward keeping a brace-depth counter until it returns to zero, and treat running off the end of the text as an error. only then split the entry's body into the key and its fields, with a second, smaller scanner that knows the three kinds of value (braced, quoted, bare). every loop needs a "reached the end" exit.

</details>

## stage 2

<details><summary>hint</summary>

one small helper per kind of rule ("non-empty string", "non-empty list of non-empty strings") keeps each rule to a line. read the pass number first, because three rules depend on it, and remember that in Python a bool is also an int.

</details>

## stage 3

<details><summary>hint</summary>

Kahn's algorithm keeps, for each paper, how many of its (distinct) references are still unread, plus the reverse map from a paper to the papers that cite it. a priority queue of ready papers gives you "alphabetically first" for free. anything left unplaced when the queue runs dry is stuck behind a cycle.

</details>

## stage 4

<details><summary>hint</summary>

each update has three parts: the jump term everyone gets, the dangling papers' total spread evenly over everyone, and each citing paper's score split over its distinct references. writing the third part with index arrays (who cites whom) keeps it vectorized. iterate a bounded number of times, stop early on a small L1 change, normalize once at the end.

</details>
