# python for numerical work · lab notes

numbers below are from an Apple-silicon laptop; yours will differ, but the ratios and shapes of the curves are typical.

## experiment 1: your dot product against NumPy's

**what people usually predict:** NumPy is "faster", maybe 5–20×, at every size. some people expect the gap to be constant, since both do the same $n$ multiplications.

**what happens:** at $n = 3$, **your `dot` wins**: about 0.13 µs against 0.2 µs for `np.dot`. they cross over around $n = 10$. after that the gap opens fast: about 10× at $n = 100$, 30× at 1,000, and **200–600× from 10,000 up**. your `dot` costs a steady ~20 ns per element at every size. `np.dot` costs a fixed ~0.2 µs per call plus well under 0.1 ns per element. (at a million elements NumPy's per-element cost rises a little again: two 8 MB arrays no longer fit in the CPU's fastest caches, so it waits on memory.) your `Vector @` is within a few percent of your `dot`, because it is your `dot` plus one method call.

**why:** your loop runs in the Python interpreter, which handles each `total += a * b` as several separate steps: fetch the next pair from `zip`, look up the type of `a`, find its multiplication routine, allocate a new float object for the product, do the same for the addition, then drop the old objects. each of those steps is cheap, but together they cost ~20 ns, which is roughly 50–100 CPU cycles for one multiply-add the hardware could do in less than one. `np.dot` makes one Python-level call, checks the arrays' types and shapes once (that's the fixed ~0.2 µs, which is why it loses at $n = 3$), then runs a compiled loop over raw 8-byte numbers, using vector instructions that do several multiply-adds per cycle. the lesson isn't "Python is slow". it's that **per-element interpreter work is slow**, so the trick (topic 02) is to hand whole arrays to compiled code and keep Python-level loops for things that happen a few times, not a million times.

## experiment 2: what is a Python number, really?

**what people usually predict:** 8 bytes per number for both, or maybe "a bit more" for the list.

**what happens:** a Python float takes **24 bytes**. the list stores an 8-byte pointer per element (8,000,056 bytes for a million), and each pointer leads to its own 24-byte float object, so the list costs about **32 bytes per number**, four times the NumPy array's 8. the float objects aren't guaranteed to sit next to each other in memory either: the median gap between their addresses is 32 bytes, but some neighbours are tens of megabytes apart (the exact numbers change from run to run), while an array's values sit exactly 8 bytes apart.

**why:** every Python value is a full object. besides the 8-byte double, a float carries a **reference count** (how many names point to it, so Python knows when to free it) and a **pointer to its type** (so `a * b` can find the right multiplication routine at run time). that's what makes Python flexible: a list can hold a float, a string and a `Vector` side by side. but it means a loop over a list chases a pointer to a scattered object, checks its type, and reads its value, for every element. a NumPy array is one block of raw doubles with a single type for all of them, so compiled code can stream through it with no checks at all. this layout, more than anything clever in the arithmetic, is where NumPy's speed comes from.

## experiment 3: just use NumPy arrays, then?

**what people usually predict:** faster, since arrays are compact and NumPy is fast. or about the same.

**what happens:** your `dot` on two NumPy arrays is **about 4× slower** than on lists (8.2 ms against 1.9 ms for 100,000 elements). the built-in `sum` is even worse: about **17× slower** on an array than on a list (3.5 ms against 0.2 ms). meanwhile `np.sum` on the same array takes 0.02 ms, about 100× faster than `sum` on the list.

**why:** a Python `for` loop can't use raw doubles. each time it takes the next element out of an array, NumPy has to **create a brand-new Python object** to hold it, a `numpy.float64` (look at the type printed under the table), and numpy scalars are slower to do arithmetic with than plain floats because their operations go through NumPy's more general machinery. a list already holds ready-made float objects, so iterating it only copies pointers. so arrays are fast only when *NumPy* does the looping. this is the most common beginner mistake with NumPy: writing `for x in arr:` and getting code that's slower than the list version. the rule for the rest of the course: **if you're writing a Python loop over the elements of an array, look for the NumPy function that does it for you**. topic 02 is a toolbox of those functions.
