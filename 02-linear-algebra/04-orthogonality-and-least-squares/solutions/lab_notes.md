# orthogonality & least squares · lab notes

## experiment 1: three ways to orthogonalize

**what people usually predict:** all three stay at machine precision (they're all "exact" algorithms), or all three degrade together as the matrix gets worse.

**what happens:** three clearly separated lines.
- **classical GS** follows the $\varepsilon\kappa^2$ guide: about $5 \times 10^{-10}$ at $\kappa = 10^4$, $10^{-5}$ at $10^6$, $2 \times 10^{-2}$ at $10^8$, and order 1 (the "orthonormal" columns are far from orthogonal) from $\kappa \approx 10^9$.
- **modified GS** follows $\varepsilon\kappa$: about $2 \times 10^{-9}$ at $\kappa = 10^8$, still only about $5 \times 10^{-5}$ at $\kappa = 10^{12}$.
- **Householder** stays flat near $10^{-15}$ for every $\kappa$.

**why:** CGS computes $q_j^\top a_l$ from the original column, so the rounding error made in computing $q_j$ (which is relative to $\|a_j\|$, not to the much smaller $\|v_j\|$) is never corrected. errors of size $\varepsilon\kappa$ in the coefficients compound to $\varepsilon\kappa^2$ in orthogonality. MGS computes each coefficient from the already-cleaned vector, so it projects out its own earlier errors and stays at $\varepsilon\kappa$. (the bounds are due to Björck, 1967, for MGS and Giraud, Langou, Rozložník & van den Eshof, 2005, for CGS.) Householder never subtracts to build $Q$. $Q$ is a product of reflections that are orthogonal by construction, so its orthogonality can't depend on $A$. the general lesson: **algorithms built from orthogonal transformations are stable**. you'll meet the same reason again for the QR algorithm (topic 05) and the SVD (topic 06).

## experiment 2: normal equations or QR?

**what people usually predict:** both methods agree to many digits until the polynomial degree gets large, then both fail together.

**what happens:** the normal equations keep 6 correct digits only up to degree 7. by degree 11 the error is ~10%, and beyond that the coefficients are meaningless. QR keeps 6+ digits all the way to degree 15. both curves track their guides: QR's error is about $\varepsilon\kappa(A)$, the normal equations' about $\varepsilon\kappa(A)^2$.

**why:** $\kappa(A^\top A) = \kappa(A)^2$ (on paper 4). forming $A^\top A$ is where the damage happens, before any solving: the cross-products round away the small singular directions of $A$. it doesn't matter how good the solver is afterwards. QR works with $A$ itself. the normal equations are still fine, and cheaper (about $mn^2$ against $2mn^2$ flops), when $A$ is well conditioned, for instance after standardizing features. ridge regression adds $\lambda I$ to $A^\top A$, which also caps the condition number (chapter 08).

## experiment 3: one point, two places

**what people usually predict:** moving any single point by 3 units moves the slope by roughly the same small amount, since it's only 1 of 21 points.

**what happens:** moving the **middle** point changes the slope by exactly **zero** (the line just shifts up by 3/21). moving the **last** point changes it by 0.39. the bar chart shows the change is a straight line in $t$: zero at the center, largest at the ends.

**why:** with centered $t$ ($\sum t_i = 0$), the least-squares slope is $\hat d = \sum_i t_i y_i / \sum_i t_i^2$, a *linear* function of $y$ with weight $t_i/\sum t_j^2$ on point $i$. a shift of $\delta$ at point $i$ changes the slope by $\delta t_i / \sum t_j^2$. here $\sum t_j^2 = 7.7$, so $3 \cdot 1/7.7 = 0.39$. points far from the center of the data (high **leverage**) control the slope. this is why a single bad measurement at an extreme input can wreck a regression, and why chapter 08 looks at leverage (the diagonal of the projection matrix $A(A^\top A)^{-1}A^\top$) and robust losses.
