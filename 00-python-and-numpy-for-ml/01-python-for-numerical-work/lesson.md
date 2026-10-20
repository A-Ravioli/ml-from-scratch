# python for numerical work

## why this matters

here's a short program that every machine learning library runs millions of times a second, written the way you might write it today:

```python
def predict(weights, x, bias):
    total = 0
    for i in range(len(weights)):
        total = total + weights[i] * x[i]
    return total + bias
```

it works. but it has three problems that this course cares about. first, nothing stops you from calling it with a 3-number `weights` and a 2-number `x`: the loop runs off the end of `x` and crashes with an `IndexError`, or, if you'd written it with `zip`, it silently ignores the extra weight and returns a wrong answer. second, the *idea* in it ("a dot product, plus a bias") is buried under index bookkeeping. the math says $w \cdot x + b$; the code says something much longer. third, it's slow, for reasons you'll measure in the lab.

ML code fixes the first two problems with ordinary Python: functions that check their inputs, and **classes** that let you write `w @ x + b` for your own objects. PyTorch's `Tensor` is a Python class, and `a + b` works on tensors because that class defines a method called `__add__`. in chapter 14 you'll write your own `Tensor` that computes gradients, using exactly this topic's trick: **operator overloading**, where `+` and `*` call methods you wrote.

so this topic tours the Python you need for numerical work: functions with type hints, classes, the "dunder" (double-underscore) methods that hook objects into Python's syntax, and dataclasses. you'll build a `Vector` class and train a tiny linear model with it. speed, the third problem, is the lab's job, and it's why topic 02 switches to NumPy.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. what does `list(zip([1, 2, 3], [10, 20]))` return?
2. you write a class `Vector` and two instances `u` and `v`. when Python evaluates `u + v`, what actually runs?
3. `3 * v` and `v * 3` look the same to a mathematician. does Python treat them the same way?
4. what does `sum([u, v, w])` compute first, before it ever touches `v` or `w`?
5. a list `a = [1, 2, 3]`, then `b = a`, then `b[0] = 99`. what is `a[0]`?

## prerequisites check

- [ ] you can write a Python function with a `for` loop and an `if` statement, and call it
- [ ] you know what a list and a tuple are, and that `x[0]` is the first element and `x[-1]` the last
- [ ] you can compute $\sqrt{3^2 + 4^2}$ in your head (the Pythagorean theorem)
- [ ] you can run `marimo edit exercise.py` from a terminal in this directory (see the repo README for setup)

if the first two feel shaky, work through chapters 1–5 and 8 (errors and exceptions) of the official [Python tutorial](https://docs.python.org/3/tutorial/) before continuing. it takes an afternoon.

### python toolkit

six pieces of Python used throughout this chapter: skim now, return when needed.

- **f-strings.** `f"got {len(u)} and {len(v)}"` puts the values of the expressions in braces into the string. `{x:.3g}` formats a number to 3 significant digits, `{x!r}` shows its `repr`.
- **exceptions.** `raise ValueError("lengths differ")` stops the function and reports an error. the caller can catch it:

  ```python
  try:
      x = risky()
  except ValueError as e:     # only this kind of error; others still propagate
      print("failed:", e)
  ```

  `except Exception` catches almost every error. you'll raise errors in topic 01 and catch them in topic 05.
- **`isinstance(obj, T)`** asks whether `obj` is of type `T` (or a subclass of it); `T` can be a tuple of types. `numbers.Real` is a ready-made type that covers `int`, `float` and NumPy's number types, so `isinstance(x, numbers.Real)` means "is a real number".
- **functions are values.** you can pass a function to another function, store it in a variable, or write a small one inline with `lambda`: `sorted(words, key=len)`, or `apply(lambda x: x ** 2, 3)`. topics 04 and 05 hand functions (a classifier, a property to test) to your code this way.
- **decorators.** a line like `@classmethod` or `@dataclass` just above a `def` or `class` hands the thing being defined to another function, which modifies or wraps it. you'll use a few ready-made ones; you won't need to write your own.
- **`with` blocks.** `with something:` runs an indented block with a setting switched on, and switches it off afterwards, even if an error happens: `with open("f.txt") as f:` closes the file at the end; `with np.errstate(divide="ignore"):` (topic 03) silences one kind of NumPy warning inside the block only.

## core ideas

### idea 1: functions with contracts

a **vector** here is just an ordered list of numbers, like $u = (1, 2, 3)$. two operations matter more than any others in machine learning. the **dot product** multiplies matching entries and adds them up:

$$u \cdot v = u_1v_1 + u_2v_2 + \dots + u_nv_n = \sum_{i=1}^{n} u_iv_i.$$

for $u = (1, 2, 3)$ and $v = (4, -5, 6)$ that's $4 - 10 + 18 = 12$. the **norm** (Euclidean length) is the Pythagorean theorem in $n$ dimensions:

$$\|u\| = \sqrt{u_1^2 + u_2^2 + \dots + u_n^2} = \sqrt{u \cdot u}.$$

$\|(3, 4)\| = \sqrt{9 + 16} = 5$. notice the second formula: the norm is built from the dot product. good code mirrors that, and `norm` should call `dot` rather than repeat its loop.

a **function** is a promise: give me inputs of this kind, and I'll return an output of that kind. Python lets you write the promise down with **type hints**:

```python
from collections.abc import Sequence

def dot(u: Sequence[float], v: Sequence[float]) -> float:
    ...
```

`Sequence[float]` means "anything you can index and take the length of, holding floats": a list or a tuple. `-> float` says what comes back. Python doesn't *enforce* hints (it runs the function either way), but they tell readers, editors and tools what you meant, and they're used throughout this course.

the part of the promise that hints can't express is what happens when the inputs are wrong. the dot product of a 3-vector and a 2-vector isn't defined. the tempting loop

```python
for a, b in zip(u, v):
    total += a * b
```

uses `zip`, which pairs elements up and **stops at the shorter input**. so `zip([1, 2, 3], [10, 20])` gives `(1, 10), (2, 20)` and quietly drops the 3 (warm-up 1). a function that silently returns a wrong number is the worst kind of bug, because nothing crashes. the fix is one line at the top: check the lengths, and `raise ValueError(...)` with a message that says what went wrong. **fail loudly, early, and with a useful message.** you'll write a lot of these checks in this course.

one more detail: in the reference, `dot([1, 2], [3, 4])` returns `11.0`, a float, even though the inputs are integers. starting the running total at `0.0` takes care of that. consistent return types make functions easier to compose.

> **▶ build: stage 1.** open the notebook with `marimo edit exercise.py` and implement `dot(u, v)` and `norm(u)` in plain Python (no NumPy yet). `dot` raises `ValueError` on a length mismatch. `norm` reuses `dot`.

### idea 2: classes, and the dunder methods that make them feel built-in

writing `dot(add(scale(2, u), v), w)` gets old fast. you'd rather write `(2 * u + v) @ w`, the way the math reads. Python allows this for your own types, through **classes**.

a class bundles data with the functions (called **methods**) that act on it. here's the skeleton of the `Vector` class you'll write:

```python
class Vector:
    def __init__(self, values):
        self.values = tuple(float(x) for x in values)
```

`Vector([1, 2, 3])` creates a new object and calls `__init__` with `self` set to that new object, so `self.values` becomes `(1.0, 2.0, 3.0)`. `self` is just the name for "the object this method was called on".

the methods whose names start and end with double underscores are called **dunder methods** (or "magic methods"). you never call them by name. Python calls them for you when you use its syntax:

| you write | Python runs |
|---|---|
| `Vector([1, 2])` | `Vector.__init__(new_object, [1, 2])` |
| `len(v)` | `v.__len__()` |
| `v[0]` | `v.__getitem__(0)` |
| `for x in v:` | `v.__iter__()`, then repeatedly asks the iterator for the next item |
| `repr(v)`, or showing `v` in a notebook | `v.__repr__()` |
| `u == v` | `u.__eq__(v)` |

define these, and your `Vector` behaves like a built-in container. `list(v)` works, because `list` just loops over `v`. a notebook shows `Vector([1.0, 2.0])` instead of the useless default `<Vector object at 0x10f3a2b50>`.

**why a tuple?** a tuple can't be changed after it's made. if `Vector` stored the list it was given, then

```python
data = [1.0, 2.0]
v = Vector(data)
data[0] = 99.0      # ...and now v has changed too!
```

because `self.values` and `data` would be two names for **the same list object**. this is warm-up 5: `b = a` doesn't copy a list, it makes `b` a second name for it, so `b[0] = 99` changes `a[0]` too. Python variables are labels attached to objects, not boxes that hold values. converting to a fresh tuple in `__init__` breaks the link, and makes every `Vector` **immutable**: once made, it never changes. immutable values are much easier to reason about, and it's how numbers already behave (`x + 1` makes a new number; it doesn't change `x`).

**equality and `NotImplemented`.** `__eq__` should compare values when both sides are vectors. but what if someone compares a vector with a list? you might be tempted to `return False`. the Python convention is subtler: **return the special value `NotImplemented`**. that means "I don't know how to compare with this type", and Python then gives the *other* object a chance to answer (by calling its `__eq__`), and only falls back to "not equal" if neither knows. returning `NotImplemented` is how independently written classes cooperate. you'll use it for every operator in stage 3.

finally, a **class method** is a method called on the class itself rather than on an instance. `Vector.zeros(3)` is a common pattern for alternative constructors:

```python
@classmethod
def zeros(cls, n):
    return cls([0.0] * n)
```

`cls` is the class, so `cls([...])` builds a new instance.

> **▶ build: stage 2.** in the `Vector` cell, implement `__init__`, `zeros`, `__len__`, `__getitem__`, `__iter__`, `__repr__` and `__eq__`.

> **🔮 predict:** once `__eq__` is defined, can you still put a `Vector` in a `set` or use it as a dictionary key? (try it after stage 2, then read the answer in "check your understanding".)

### idea 3: operator overloading

now the payoff. Python translates arithmetic into dunder calls too:

| you write | Python runs |
|---|---|
| `u + v` | `u.__add__(v)` |
| `u - v` | `u.__sub__(v)` |
| `-u` | `u.__neg__()` |
| `v * 3` | `v.__mul__(3)` |
| `3 * v` | `int.__mul__(3, v)` first, and if that returns `NotImplemented`, then `v.__rmul__(3)` |
| `v / 2` | `v.__truediv__(2)` |
| `u @ v` | `u.__matmul__(v)` |
| `abs(v)` | `v.__abs__()` |

that answers warm-up 2: `u + v` is a method call, `u.__add__(v)`, and the method can do whatever you program it to do. for a vector, it should build a **new** `Vector` of componentwise sums and leave `u` and `v` alone.

the `3 * v` row answers warm-up 3. Python always asks the **left** operand first. the integer `3` has never heard of your `Vector`, so `int.__mul__` returns `NotImplemented`. Python then tries the **reflected** method on the right operand, `v.__rmul__(3)`. if you don't define `__rmul__`, `3 * v` raises `TypeError` even though `v * 3` works. since multiplying by a number commutes, `__rmul__` can simply call `__mul__`.

`@` is Python's matrix-multiplication operator (NumPy uses it for exactly that). for two vectors, the natural meaning is the dot product, so `u @ v` returns a float. what about `u * v` for two vectors? mathematicians disagree about what it should mean (NumPy says elementwise; others say it's undefined), so this class refuses: `__mul__` returns `NotImplemented` for anything that isn't a number, and Python raises `TypeError`. **refusing an ambiguous operation is better than guessing.**

there's one charming corner left: `sum`. the built-in `sum(items)` starts from the number `0` and adds each item in turn, so the very first thing it computes is `0 + items[0]` (warm-up 4). Python asks `int.__add__(0, v)`, gets `NotImplemented`, and then calls `v.__radd__(0)`. if your `__radd__` treats `0` as the zero vector and returns `v`, then `sum(list_of_vectors)` just works. you'll use that in stage 4 to add up gradients.

here's how a couple of these look in the reference:

```python
def __add__(self, other):
    if not isinstance(other, Vector):
        return NotImplemented
    if len(self) != len(other):
        raise ValueError(...)
    return Vector(a + b for a, b in zip(self.values, other.values))
```

`a + b for a, b in zip(...)` is a **generator expression**: a lazy loop that produces values one at a time, which `Vector.__init__` happily consumes, since it accepts any iterable.

> **▶ build: stage 3.** add `__add__`, `__radd__`, `__neg__`, `__sub__`, `__mul__`, `__rmul__`, `__truediv__`, `__matmul__` and `__abs__` to `Vector`. reuse your stage 1 functions for `@` and `abs`.

> **🔮 predict:** how many new `Vector` objects does evaluating `2 * u - v / 2 + -u` create? (count them, then think about what that costs for vectors with a million entries.)

### idea 4: dataclasses, callable objects, and a first training step

many classes are mostly a bundle of named fields: a model's weights and bias, a training configuration. writing `__init__`, `__repr__` and `__eq__` for each by hand is tedious. the `@dataclass` decorator writes them for you from a list of annotated fields:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LinearModel:
    weights: Vector
    bias: float = 0.0
```

that's a complete class. `LinearModel(Vector([1, 2]), 0.5)` works, prints as `LinearModel(weights=Vector([1.0, 2.0]), bias=0.5)`, and compares field by field. `bias: float = 0.0` gives a default. `frozen=True` makes instances immutable: `model.bias = 1.0` raises `FrozenInstanceError`. to get a modified copy you use `dataclasses.replace(model, bias=1.0)`, which builds a new instance with one field changed.

one more dunder: `__call__`. defining it makes instances **callable** like functions, so `model(x)` runs `model.__call__(x)`. that's exactly how PyTorch models are used (`logits = net(images)`), and it reads naturally: a model *is* a function from inputs to predictions,

$$\hat y = f(x) = w \cdot x + b.$$

how good is a model? on $n$ examples $(x_1, y_1), \dots, (x_n, y_n)$, the **mean squared error** averages the squared mistakes:

$$\text{MSE}(w, b) = \frac{1}{n}\sum_{i=1}^{n}\big(f(x_i) - y_i\big)^2.$$

squaring makes every mistake count as positive and punishes big ones more than small ones.

how do you make a model better? you don't need calculus yet to follow the idea (chapters 01 and 03 derive it properly). write $e_i = f(x_i) - y_i$ for the error on example $i$. if $e_i > 0$, the prediction was too big, so the bias should go down a little, and each weight $w_j$ should move against the sign of $x_{ij}$. summing these nudges over all examples, scaled by $2/n$ (it comes from differentiating the square), gives the **gradient**:

$$g_w = \frac{2}{n}\sum_{i=1}^{n} e_i\,x_i, \qquad g_b = \frac{2}{n}\sum_{i=1}^{n} e_i.$$

the gradient points in the direction that makes the error grow fastest, so one step of **gradient descent** moves the other way, by a small amount set by the **learning rate** $\eta$:

$$w \leftarrow w - \eta\,g_w, \qquad b \leftarrow b - \eta\,g_b.$$

with your `Vector`, the code for $g_w$ is almost the formula: `(2 / n) * sum(e * x for e, x in zip(errors, xs))`. that line uses `__rmul__` (a float times a vector), `__radd__` (inside `sum`) and `__add__`. repeat the step a few hundred times and the model fits the data. that loop, predict, measure the error, compute the gradient, step, is how every neural network in this course trains. in chapter 14 the only difference is that the gradient is computed for you.

> **▶ build: stage 4.** implement `LinearModel.__call__`, `mse` and `step`. `step` returns a new model; the one you called it on stays unchanged.

## worked example

take two training examples in two dimensions, a model that starts at zero, and learning rate $\eta = 0.1$:

$$x_1 = (1, 0),\ y_1 = 1; \qquad x_2 = (0, 2),\ y_2 = 2; \qquad w = (0, 0),\ b = 0.$$

**predictions and errors.** $f(x_1) = 0\cdot 1 + 0\cdot 0 + 0 = 0$ and $f(x_2) = 0$. so $e_1 = 0 - 1 = -1$ and $e_2 = 0 - 2 = -2$.

**loss.** $\text{MSE} = \tfrac12\big((-1)^2 + (-2)^2\big) = \tfrac52 = 2.5$.

**gradient.** with $n = 2$, the factor $2/n$ is 1:

$$g_w = 1\cdot\big((-1)(1, 0) + (-2)(0, 2)\big) = (-1, 0) + (0, -4) = (-1, -4), \qquad g_b = (-1) + (-2) = -3.$$

**step.** $w \leftarrow (0, 0) - 0.1\,(-1, -4) = (0.1, 0.4)$ and $b \leftarrow 0 - 0.1\,(-3) = 0.3$.

**check.** the new predictions are $f(x_1) = 0.1 + 0.3 = 0.4$ and $f(x_2) = 0.8 + 0.3 = 1.1$. the errors are now $-0.6$ and $-0.9$, so the loss is $\tfrac12(0.36 + 0.81) = 0.585$, down from 2.5. ✓

the tests check exactly these numbers: your `step` must return weights $(0.1, 0.4)$ and bias $0.3$. if you get $(0.05, 0.2)$ you've lost the factor 2. if you get $(-0.1, -0.4)$ you've stepped uphill.

## on paper

1. **the norm of a sum.** using only the definition $\|u\|^2 = u \cdot u$ and the fact that the dot product distributes ($u \cdot (v + w) = u\cdot v + u \cdot w$), show that $\|u + v\|^2 = \|u\|^2 + 2\,u\cdot v + \|v\|^2$. when is $\|u + v\|^2 = \|u\|^2 + \|v\|^2$?

<details><summary>solution</summary>

$\|u + v\|^2 = (u + v)\cdot(u + v) = u\cdot u + u \cdot v + v \cdot u + v \cdot v = \|u\|^2 + 2\,u\cdot v + \|v\|^2$, since $u \cdot v = v \cdot u$. the cross term vanishes exactly when $u \cdot v = 0$, which is when $u$ and $v$ are perpendicular: that's the Pythagorean theorem. (you'll meet this identity again in topic 02, where it makes distance computations fast.)

</details>

2. **one step by hand.** a one-dimensional model $f(x) = wx + b$ starts at $w = 1, b = 0$. the data is $(x, y) = (1, 3)$ and $(2, 5)$. compute the MSE, the gradient, and the model after one step with $\eta = 0.1$. did the loss go down?

<details><summary>solution</summary>

predictions $1$ and $2$, errors $e = (-2, -3)$, MSE $= \tfrac12(4 + 9) = 6.5$. with $2/n = 1$: $g_w = (-2)(1) + (-3)(2) = -8$ and $g_b = -5$. the new model is $w = 1 + 0.8 = 1.8$ and $b = 0.5$. predictions become $2.3$ and $4.1$, errors $-0.7$ and $-0.9$, MSE $= \tfrac12(0.49 + 0.81) = 0.65$. yes, a tenfold drop.

</details>

## experiment

`lab.py` asks one question three ways: **how fast is all this?** you'll time your `dot` and your `Vector` against the same operations in NumPy, across a range of sizes, then look at what a Python number actually is in memory, and at what happens when you loop over a NumPy array with a Python `for`. predict first.

## check your understanding

1. why does `dot` check the lengths explicitly instead of relying on the loop to crash?

<details><summary>answer</summary>

because a `zip` loop doesn't crash: it stops at the shorter input and returns a plausible-looking wrong number. an index-based loop might crash, or might not (if `u` is the shorter one). an explicit check turns a silent wrong answer into an immediate, clearly described error.

</details>

2. what would go wrong if `Vector.__init__` did `self.values = values`?

<details><summary>answer</summary>

the vector would share the caller's list. changing the list later would change the vector (aliasing), and the vector wouldn't be immutable. it would also keep whatever type was passed in: a generator would be used up after one loop, and integers would stay integers.

</details>

3. (the predict callout in idea 2.) can you put a `Vector` in a set after defining `__eq__`?

<details><summary>answer</summary>

no: `TypeError: unhashable type`. when a class defines `__eq__` without `__hash__`, Python sets `__hash__` to `None`, because objects that compare equal must have equal hashes, and the default hash (based on identity) would break that. since your `Vector` is immutable, you could safely add `def __hash__(self): return hash(self.values)`. frozen dataclasses do this automatically.

</details>

4. why should `__add__` return `NotImplemented` for a non-vector, rather than raising `TypeError` itself?

<details><summary>answer</summary>

so the other operand gets a turn. if someone writes a `Matrix` class that knows how to add itself to a `Vector`, `v + m` should be able to fall back to `m.__radd__(v)`. returning `NotImplemented` allows that; raising immediately doesn't. if nobody can handle it, Python raises the `TypeError` for you.

</details>

5. (the predict callout in idea 3.) how many new vectors does `2 * u - v / 2 + -u` create?

<details><summary>answer</summary>

`2 * u` (one), `v / 2` (two), the subtraction (three, plus one more inside `__sub__`, since it computes `self + (-other)`: four), `-u` (five), and the final `+` (six). every intermediate result is a full new object, with every component converted and stored. for long vectors, these temporaries cost real time and memory. NumPy has the same issue (each operation allocates a new array), which is why it offers in-place operators like `+=`, and why libraries like PyTorch fuse operations together.

</details>

## common bugs

- **`zip` hides a length mismatch.** symptom: `dot([1, 2, 3], [1, 2])` returns 5 instead of raising. fix: compare `len(u)` and `len(v)` before looping.
- **returning `False` (or raising) instead of `NotImplemented`.** symptom: comparisons and arithmetic with other types behave oddly, and `3 * v` fails even though you defined `__rmul__` on some other class. fix: `return NotImplemented` whenever the other operand isn't a type you handle.
- **forgetting `__rmul__` or `__radd__`.** symptom: `v * 3` works but `3 * v` raises `TypeError`; `u + v` works but `sum([u, v])` raises `TypeError: unsupported operand type(s) for +: 'int' and 'Vector'`.
- **modifying `self` in an operator.** symptom: after `w = u + v`, `u` has changed too. operators must build and return a new object.
- **storing the caller's list.** symptom: a vector changes when you modify the list you built it from. fix: `tuple(float(x) for x in values)`.
- **the gradient step goes uphill.** symptom: the loss grows every step. you added `lr * gradient` instead of subtracting it, or you computed the error as `y - prediction` instead of `prediction - y`.

## going deeper

- the official Python tutorial, [chapter 9: classes](https://docs.python.org/3/tutorial/classes.html), and the [data model reference](https://docs.python.org/3/reference/datamodel.html), which lists every dunder method and exactly when Python calls it (see "emulating numeric types" for the reflected-operator rules).
- Ramalho (2022), *Fluent Python*, 2nd ed., O'Reilly, chapters 11–12 and 16: a Pythonic vector class built step by step, and operator overloading in depth.
- [PEP 465](https://peps.python.org/pep-0465/) (2014), which added the `@` operator to Python specifically for numerical code, with a clear explanation of why.
- [PEP 557](https://peps.python.org/pep-0557/) (2017), the design of dataclasses, and [PEP 484](https://peps.python.org/pep-0484/) (2014), type hints.
- Karpathy (2020), [micrograd](https://github.com/karpathy/micrograd): a tiny autograd engine whose `Value` class is a `Vector`-like class with operator overloading that also tracks gradients. you'll build your own in chapter 14.

## connections

- ← builds on: basic Python (functions, loops, lists).
- → used in: NumPy arrays, which are a much faster version of your `Vector` (topic 02); `assert`-based checks and testing (topic 05); the dot product and norm throughout linear algebra (ch 01, ch 02); the predict → loss → gradient → step loop in optimization (ch 06) and linear models (ch 08); the `Value` and `Tensor` classes with operator overloading in your autograd engine (ch 14).
