# structural causal models & do-calculus · hints

open a hint only after you've been stuck for a while. this elective is spec-driven: there's one hint per stage, and it points at the idea, not the code.

## stage 1

<details><summary>hint</summary>

for `topological_order`, keep a set of placed nodes and sweep over the unplaced ones, placing any node whose parents are all placed. if a whole sweep places nothing while nodes remain, there's a cycle: raise instead of sweeping again. validate up front that every parent is a key. for `ancestors`, do a depth-first search that follows `parents[v]`; for `descendants`, build the children lists once (`{v: [c for c in parents if v in parents[c]]}`) and search along them. start the search from the neighbours, not from the node, so the node itself isn't included.

</details>

## stage 2

<details><summary>hint</summary>

the order of operations is what the checks test: `noise = as_rng(rng).normal(size=(n, len(self.nodes)))` first, then a loop over `self.order`. for node `v` with column `k = self.nodes.index(v)`, either use the intervention value (`np.broadcast_to(np.asarray(do[v], float), (n,)).copy()`) or call `self.mechanisms[v]({p: values[p] for p in self.parents[v]}, noise[:, k])` and broadcast that to `(n,)` too. because the noise matrix doesn't depend on `do`, two calls with the same seed describe the same units in two worlds.

</details>

## stage 3

<details><summary>hint</summary>

the moral-graph test is the least fiddly: collect `xs | ys | zs` and all their ancestors; for every kept node add undirected edges to each of its kept parents and between every pair of its kept parents; then run a graph search from `xs` that never steps onto a node in `zs`, and report `False` the moment it reaches `ys`. for the backdoor test, return `False` if `zs` meets `descendants(parents, x)`; otherwise delete `x` from every parents list (that removes the edges *out of* `x`) and call `d_separated`.

</details>

## stage 4

<details><summary>hint</summary>

stack the `zs` columns into a matrix and get stratum labels with `np.unique(Z, axis=0, return_inverse=True)` (call `.ravel()` on the labels: some NumPy versions return them 2-D). for each stratum `s`, the weight is `np.mean(label == s)` over all rows and the value is `y[(label == s) & (x == x_value)].mean()`; raise if that cell is empty. `frontdoor_adjust` is two nested loops of the same kind: over the values of `m` seen among the treated rows (weight: their share of the treated rows), then over every value `x'` of `x` (weight `np.mean(x == x')`, value `y[(m == m_val) & (x == x')].mean()`).

</details>
