# multivariable calculus · lab notes

## experiment 1: gradients and level sets

**what people usually predict:** most people know the textbook answer ("perpendicular"), but many expect it to hold only roughly on a wiggly function, or only near minima. on the second question, a common guess is that tightly packed contours mean a *small* gradient ("lots of lines, lots of detail, small steps").

**what happens:** over about 540 points on 12 contour lines, the angle between the curve and your steepest-ascent direction averages **89.7°**, with the worst case around **87.5°**. the long arrows sit exactly where contour lines crowd together; near the minimum and the saddles, where the lines spread out, the arrows shrink to nothing.

**why:** along a level curve $f$ is constant, so its derivative along the tangent $t$ is zero, and $D_tf = \nabla f\cdot t = 0$. that's an exact statement about every differentiable function, wiggly or not. the residual error of a degree or two comes from the *contour*, not the gradient: matplotlib draws level sets by linear interpolation on a $241 \times 241$ grid, so its polyline tangents are slightly off where the curves bend sharply. for spacing: neighbouring contours differ by a fixed $\Delta f$, and moving perpendicular to them, $\Delta f \approx \|\nabla f\|\cdot\text{spacing}$. so the spacing is inversely proportional to the slope. a contour map is a picture of the gradient field: direction from the normals, size from the density.

## experiment 2: does "steepest" depend on your units?

**what people usually predict:** that changing units can't change anything real: same function, same point, same downhill. so the two directions should agree, and both gradient descents should reach $(1, 1)$.

**what happens:** with $s = 10$, steepest descent in $(x, y)$ points at the minimum, $(0.707, 0.707)$. steepest descent in $(x, u)$, mapped back to $(x, y)$, is $(1.000, 0.010)$: **44.4° apart**, almost purely along $x$. after 25 steps, $(x, y)$-gradient descent is at $(0.992, 0.992)$, while $(x, u)$-gradient descent has fixed $x$ but barely moved in $y$: it's at $(0.992, -0.902)$. Newton's direction $-H^{-1}\nabla f$, computed from your `numerical_hessian` in each system, agrees to about $10^{-6}$ degrees.

**why:** with $y = u/s$, the chain rule gives $\partial g/\partial u = (\partial f/\partial y)/s$: the gradient's $u$-component shrinks by $s$. a step $\Delta u$ in that direction moves $y$ by only $\Delta u/s$, another factor of $s$. so in physical terms the $y$-motion is suppressed by $s^2 = 100$ relative to $x$. "steepest" means "largest decrease per unit step length", and the change of units changed what a unit step length *is*: one unit of $u$ is only a tenth of a unit of $y$. the gradient isn't a property of the function alone; it's the function plus a choice of inner product on the inputs. Newton's step, by contrast, is "jump to the minimum of the local quadratic model", which is a geometric description that doesn't mention lengths at all: the Hessian picks up a factor $1/s^2$ that exactly cancels. this is why gradient descent is sensitive to feature scaling (chapter 07's standardization), why ill-conditioned problems zig-zag (chapter 06), and why preconditioning and Newton-like methods help. the capstone builds the Newton solver.

## experiment 3: how big should the Hessian's step be?

**what people usually predict:** that the step that's best for gradients ($10^{-6}$, or $10^{-5}$) is also best for Hessians, with similar accuracy, around $10^{-10}$.

**what happens:** the second-difference curve is a U with its bottom near **$h \approx 3 \times 10^{-4}$** and relative error about **$10^{-8}$**. at $h = 10^{-6}$ the error has climbed to about **$5 \times 10^{-4}$**: three or four correct digits. differencing the *exact* gradient with your `numerical_jacobian` does far better: best near $h = 10^{-5}$, with error about **$3 \times 10^{-11}$**.

**why:** second differences divide a sum of four nearly cancelling function values by $h^2$. the truncation error is $O(h^2)$, the rounding error is about $\varepsilon|f|/h^2$, and they balance at $h \sim \varepsilon^{1/4} \approx 10^{-4}$, where both are around $\sqrt\varepsilon \approx 10^{-8}$. at $h = 10^{-6}$ the rounding term is $10^{-16}/10^{-12} = 10^{-4}$. differencing an exact gradient is a *first* difference, governed by $h^2 + \varepsilon/h$: best at $\varepsilon^{1/3} \approx 10^{-5}$ with error $\varepsilon^{2/3} \approx 10^{-11}$. the practical rule: if you can get exact gradients (topic 04's forward mode, chapter 14's backprop), build Hessians by differencing them, or better, compute them exactly too; the capstone does the latter with nested dual numbers. second differences of $f$ alone are the last resort, and need $h \approx 10^{-4}$.
