# scalar reverse-mode autodiff

## why this matters

in topic 01 you derived the gradients of a one-hidden-layer network by hand: six lines of chain rule, each one a chance for a transposed matrix or a forgotten factor, all of it to be redone the moment you add a layer, change an activation, or try a different loss. real networks have hundreds of layers and are changed daily. nobody derives their gradients by hand; a program does it, exactly (not approximately, like finite differences) and at a cost of a few forward passes, however many parameters there are.

that program is **reverse-mode automatic differentiation**, and its core is surprisingly small. every number in your computation remembers which operation produced it and from which inputs, so the computation leaves behind a graph. each operation knows one local fact: its own derivative. one sweep backwards through the graph, applying the chain rule at every node, then gives the derivative of the output with respect to every input at once. in chapter 03 you built *forward*-mode autodiff with dual numbers, which pushes derivatives forward alongside values and needs one pass per input; reverse mode needs one pass in total, for any number of inputs. that's why it trains neural networks. you'll build it for scalars here, about a hundred lines in the style of Karpathy's micrograd, and topic 03 scales the same design to arrays.

## warm-up: predict first

1. $c = a \cdot b$. if you know $\partial L/\partial c$, what are $\partial L/\partial a$ and $\partial L/\partial b$?
2. $y = x \cdot x$, computed as one multiplication node whose two inputs are both $x$. what should $x$'s gradient be, and what goes wrong if each input "sets" its gradient instead of adding to it?
3. why must the backward pass visit nodes in a particular order? what order?
4. $f: \mathbb R^n \to \mathbb R$. how many passes does forward mode need for the full gradient? reverse mode?
5. you call `backward()` twice on the same loss. what should happen to the gradients of the inputs?

## prerequisites check

- [ ] you can apply the chain rule to a composition of several functions (→ ch 01 · topic 02)
- [ ] you've built forward-mode autodiff with `Dual` numbers (→ ch 03 · topic 04)
- [ ] you can derive the gradients of a small network by hand (→ ch 14 · topic 01)
- [ ] you know what a graph, a DAG and depth-first search are (any intro to algorithms; see "going deeper")
- [ ] you can check a gradient with finite differences (→ ch 02 · topic 07)

## core ideas

### idea 1: values that remember where they came from

a `Value` wraps one float, `.data`, and adds three things: `.grad`, which will hold $\partial L/\partial(\text{this value})$ for the final output $L$; `._prev`, the tuple of Values it was computed from (empty for inputs you create); and `._op`, a label like `"*"`. arithmetic on Values returns new Values that record their parents, so evaluating an expression builds a **computational graph**: a directed acyclic graph whose leaves are the inputs and whose root is the output. a plain number in an expression (`a + 1.5`) is wrapped as a constant `Value`, so every operand is a node.

each node also carries a closure, `._backward`, that knows the operation's **local derivative** and applies the chain rule one step: given `out.grad` $= \partial L/\partial\,\text{out}$, add each parent's share. for $\text{out} = a + b$, $\partial\,\text{out}/\partial a = \partial\,\text{out}/\partial b = 1$, so both parents get `out.grad`. for $\text{out} = a \cdot b$, the local derivatives are $b$ and $a$ (warm-up 1):

$$\frac{\partial L}{\partial a} \mathrel{+}= b\,\frac{\partial L}{\partial\,\text{out}},\qquad \frac{\partial L}{\partial b} \mathrel{+}= a\,\frac{\partial L}{\partial\,\text{out}}.$$

the `+=` is not a detail. a value can feed several operations (or the same one twice, as in $x \cdot x$), and the multivariable chain rule says its gradient is the **sum** over all the paths it influences $L$ through: $\partial L/\partial v = \sum_{\text{children } u}(\partial L/\partial u)(\partial u/\partial v)$. if each use *assigned* its contribution, the last one would overwrite the others, and $x \cdot x$ would get gradient $x$ instead of $2x$ (warm-up 2).

> **▶ build: stage 1.** implement `Value.__init__`, `__add__` and `__mul__`, each recording `(self, other)` and its op, and setting `out._backward` to a closure that adds into the parents' `.grad`. (`__radd__`, `__rmul__` and `__repr__` are given.)

### idea 2: a small library of local rules

every other operation is the same pattern with a different local derivative. some are compositions you get for free: $-a = a \cdot (-1)$, $a - b = a + (-b)$, $a / b = a \cdot b^{-1}$. the rest need their own rule:

| op | value | local derivative |
|---|---|---|
| $a^p$ ($p$ a number) | $a^p$ | $p\,a^{p-1}$ |
| $\exp a$ | $e^a$ | $e^a$ (the output itself) |
| $\log a$ | $\log a$ | $1/a$ |
| $\tanh a$ | $t = \tanh a$ | $1 - t^2$ |
| $\operatorname{relu} a$ | $\max(a, 0)$ | $1$ if $a > 0$, else $0$ |

notice how many derivatives are cheapest in terms of the *output* ($e^a$, $1 - t^2$): the closure captures the computed value, so the backward pass doesn't recompute it. a Value exponent ($a^b$) isn't supported: raise `TypeError` (you could add it as $\exp(b\log a)$).

compare this with chapter 03's `Dual`. there, each operation computed a value and pushed a *tangent* forward: $(a + \dot a\varepsilon)(b + \dot b\varepsilon) = ab + (\dot ab + a\dot b)\varepsilon$. here, each operation computes a value and *stores a recipe* for pulling an *adjoint* backward later. same local derivatives, opposite direction.

> **▶ build: stage 2.** implement `__neg__`, `__sub__`, `__pow__`, `__truediv__`, `exp`, `log`, `tanh` and `relu`.

### idea 3: the backward pass

with every local rule in place, computing all the gradients is a matter of calling the `_backward` closures in the right order. a node may pass its gradient to its parents only once its *own* gradient is complete, i.e. after every node that uses it has added its contribution (warm-up 3). so:

1. list the graph's nodes in **topological order**: every node after all of its parents. depth-first search does it: visit a node's parents first, then append the node.
2. set the root's gradient to $\partial L/\partial L = 1$.
3. walk the list **backwards**, calling each node's `_backward`. by the time a node is reached, every node that uses it (all of which come later in the list) has already added its contribution.

two details. **no recursion**: a recursive depth-first search uses one Python stack frame per step along a path, and Python's default limit is 1,000 frames, so a graph that's merely *long* (a loop that applies 5,000 operations) crashes. use an explicit stack of `(node, expanded)` pairs instead: pop a node; if it's not expanded, push it back as expanded and push its parents; when an expanded node is popped, all its parents are done, so append it. a `seen` set stops shared nodes from being expanded twice. **repeated calls**: before sweeping, reset the gradients of every *intermediate* node to 0 (their old values belong to the last sweep), but leave the leaves alone, so that calling `backward()` twice adds the two gradients, as PyTorch does (warm-up 5).

> **🔮 predict:** in $L = \tanh(a b + a)$ with $a = 2$, $b = -3$, $a$ is used twice. which of its two paths to $L$ contributes more to $\partial L/\partial a$, and what's the sign of the total?

> **▶ build: stage 3.** implement `topological_order(root)` (iteratively) and `Value.backward()`.

### idea 4: reverse versus forward, and what it costs

for $f: \mathbb R^n \to \mathbb R$, forward mode (your `Dual`) gets one directional derivative per pass, so the gradient takes $n$ passes. reverse mode records the graph during one forward pass and recovers every partial derivative in one backward pass, whose cost is a small constant times the forward pass (each node's `_backward` does about as much work as its forward operation). so the gradient costs $O(1)$ forward passes instead of $O(n)$ (warm-up 4). the price is memory: the whole graph must be kept until the backward pass is done (topic 06 measures and reduces it).

`value_and_grad(f, x)` packages this: wrap each $x_i$ as a `Value`, evaluate `f` on the list, call `backward()`, and return the value and the inputs' gradients as an array. `graph_size(root)` counts the nodes, which is what the time and memory of both passes are proportional to.

> **🔮 predict:** a function of $n = 300$ inputs whose graph has about $6n$ nodes. roughly how many times faster will your reverse mode compute its gradient than your forward-mode `grad` from chapter 03? (on paper 4 makes the estimate.)

> **▶ build: stage 4.** implement `value_and_grad(f, x)` and `graph_size(root)`.

## worked example

$L = \tanh(ab + a)$ at $a = 2$, $b = -3$. the graph has five nodes: $a$, $b$, $c = a \cdot b$, $d = c + a$, $L = \tanh d$.

**forward.** $c = -6$, $d = -6 + 2 = -4$, $L = \tanh(-4) = -0.999329$.

**backward**, in reverse topological order $L, d, c, b, a$ (or $L, d, c, a, b$):

- $L$: $\bar L = 1$. its `_backward` gives $\bar d = 1 - \tanh^2(-4) = 0.0013410$.
- $d = c + a$: both parents get $\bar d$: $\bar c = 0.0013410$, and $a$ receives its first contribution, $0.0013410$.
- $c = a \cdot b$: $\bar a \mathrel{+}= b\,\bar c = -3 \times 0.0013410 = -0.0040229$, and $\bar b \mathrel{+}= a\,\bar c = 2 \times 0.0013410 = 0.0026819$.

so $\partial L/\partial a = 0.0013410 - 0.0040229 = -0.0026819 = (1 - \tanh^2(-4))(b + 1)$ and $\partial L/\partial b = 0.0026819 = (1 - \tanh^2(-4))\,a$. the path through the product contributes three times as much as the direct path, with the opposite sign (the predict callout). note how small the gradients are: $\tanh$ is saturated at $-4$, so its derivative, $0.0013$, scales everything behind it. (saturating activations shrinking gradients is a theme of chapter 15.) the tests check every number here.

## on paper

1. **the chain rule on a graph.** for $L = f(u, v)$ with $u = g(x)$ and $v = h(x)$, show that $\partial L/\partial x = (\partial L/\partial u)g'(x) + (\partial L/\partial v)h'(x)$, and explain why this forces `+=` in every `_backward`.

<details><summary>solution</summary>

$L(x) = f(g(x), h(x))$. the multivariable chain rule gives $\frac{dL}{dx} = \frac{\partial f}{\partial u}\frac{du}{dx} + \frac{\partial f}{\partial v}\frac{dv}{dx}$: one term per path from $x$ to $L$. in the graph, $x$ has two children, and each child's `_backward` computes one term. the gradient of $x$ is their sum, so each child must add its term to whatever is already there.

</details>

2. **division without its own rule.** using only the rules for $\cdot$ and $a^{-1}$, derive $\partial(a/b)/\partial a$ and $\partial(a/b)/\partial b$, and check them against the quotient rule.

<details><summary>solution</summary>

$a/b = a \cdot w$ with $w = b^{-1}$. the product gives $\partial/\partial a = w = 1/b$ and $\partial/\partial w = a$; the power rule gives $\partial w/\partial b = -b^{-2}$, so $\partial/\partial b = a \cdot (-b^{-2}) = -a/b^2$. the quotient rule for $a/b$ gives the same: $\partial_a = 1/b$, $\partial_b = -a/b^2$.

</details>

3. **why a topological order works.** prove that if nodes are processed in reverse topological order, each node's gradient is complete when its `_backward` runs.

<details><summary>solution</summary>

a node $v$'s gradient receives contributions only from the `_backward`s of its children (the nodes that have $v$ as a parent). in a topological order every child comes after $v$, so in the reversed order every child is processed before $v$. by the time $v$ is reached, all its contributions have been added, and nothing will add to it later.

</details>

4. **cost of forward versus reverse.** $f$ has $n$ inputs and its graph has $N$ nodes. how many node operations do forward and reverse mode perform for the full gradient? for experiment 1's $f$ ($N \approx 6n$), what are they at $n = 300$?

<details><summary>solution</summary>

forward: $n$ passes over the graph, each touching every node: about $nN$ operations, $O(n^2)$ when $N \propto n$. reverse: one forward pass and one backward pass: about $2N$. at $n = 300$: $300 \times 1800 = 540{,}000$ against $3{,}600$, a factor of 150. (your `Dual` passes are also slower per operation, which the lab sees too.)

</details>

## experiment

`lab.py` runs two experiments:

1. the gradient of one function of $n$ inputs by your reverse-mode `value_and_grad` and by your forward-mode `grad` from chapter 03, for $n$ from 10 to 300.
2. one dense layer ($784 \to 32$) on one image, as a graph of scalar `Value`s, against the same computation in NumPy.

## check your understanding

1. what does a leaf's `.grad` contain after `L.backward()`?

<details><summary>answer</summary>

$\partial L/\partial(\text{leaf})$, the total derivative through every path from that leaf to $L$ (plus whatever was there before, if you didn't reset it).

</details>

2. why does `__add__` wrap a plain number as a `Value` instead of just adding it to `.data`?

<details><summary>answer</summary>

so that every operand is a node with a `.grad` slot and the closure can treat both parents the same way. it's slightly wasteful (the constant's gradient is computed and never used), but it keeps every rule uniform. topic 03's engine instead marks constants with `requires_grad=False` and skips them.

</details>

3. in the stack-based topological sort, why push a node back as "expanded" before pushing its parents, instead of appending it right away?

<details><summary>answer</summary>

a node may only be appended after all of its parents. pushing it back as expanded means it's popped again only after everything pushed above it (its parents and, recursively, their ancestors) has been processed and appended.

</details>

4. `backward()` resets the intermediate nodes' gradients but not the leaves'. why both halves of that rule?

<details><summary>answer</summary>

intermediate gradients belong to one sweep: leftovers from a previous call would be added again and double-count every path through them. leaf gradients are what you're accumulating: summing gradients over several backward calls (several losses, or several mini-batches) is useful, and optimizers reset them explicitly with `zero_grad`.

</details>

5. when would forward mode be the better choice?

<details><summary>answer</summary>

when there are few inputs and many outputs (a Jacobian of a function $\mathbb R \to \mathbb R^{1000}$, or a sensitivity with respect to one parameter of a simulation), or when memory is the constraint: forward mode needs no graph. topic 06 makes the comparison precise.

</details>

6. relu's derivative at exactly 0 is set to 0. does it matter?

<details><summary>answer</summary>

the derivative doesn't exist there, so any value in $[0, 1]$ is a valid subgradient. with real-valued inputs, landing exactly on 0 is rare, so it almost never matters; 0 is the convention PyTorch uses, which keeps results comparable.

</details>

## common bugs

- **`=` instead of `+=` in a `_backward`.** symptom: correct gradients for expressions where every value is used once; wrong ones for $x \cdot x$, $a + a$, or any shared subexpression.
- **forgetting `out.grad` in the local rule.** symptom: gradients that are right only when the node is the root (where `out.grad` is 1).
- **visiting nodes in the wrong order.** symptom: some gradients are 0 or too small, because a node passed its gradient on before all of its contributions had arrived.
- **recursion.** symptom: `RecursionError` for long chains, correct results for short ones.
- **not resetting intermediate grads.** symptom: calling `backward()` twice gives more than twice the gradient.
- **using `math` functions that don't know about Values.** symptom: `TypeError: must be real number, not Value`. inside the engine, apply `math.exp` to `.data`; outside it, call the method `v.exp()`.

## going deeper

- Karpathy (2020), *micrograd*: a tiny scalar autograd engine and neural network library, the model for this topic. [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
- Linnainmaa (1976), "Taylor expansion of the accumulated rounding error", *BIT Numerical Mathematics* 16: 146–160. the first description of reverse-mode differentiation of a computer program. [doi:10.1007/BF01931367](https://doi.org/10.1007/BF01931367)
- Griewank & Walther (2008), *Evaluating Derivatives*, 2nd ed., SIAM, chapter 3 (the reverse mode). [doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
- Baydin, Pearlmutter, Radul & Siskind (2018), "Automatic differentiation in machine learning: a survey", *JMLR* 18(153): 1–43. [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
- Cormen, Leiserson, Rivest & Stein (2022), *Introduction to Algorithms*, 4th ed., MIT Press, §20.4 (topological sort by depth-first search).

## connections

- ← builds on: the chain rule (ch 01 · topic 02), forward-mode `Dual` (ch 03 · topic 04), hand-derived network gradients (topic 01), gradient checks (ch 02 · topic 07).
- → used in: the array version of the same engine, `mlfs.tensor` (topic 03), VJPs and memory (topic 06), and every gradient computed in the rest of the course.
