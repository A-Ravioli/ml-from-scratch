# Neural Ordinary Differential Equations (Neural ODEs)

## Prerequisites
- Ordinary differential equations and dynamical systems
- Numerical integration methods (Euler, Runge-Kutta)
- Backpropagation and automatic differentiation
- ResNet and continuous depth concepts

## Learning Objectives
- Master the connection between ResNets and continuous dynamics
- Understand Neural ODE formulation and adjoint sensitivity method
- Implement adaptive solvers for forward and backward passes
- Analyze computational trade-offs and memory efficiency
- Connect to modern continuous-depth architectures

## Mathematical Foundations

### 1. From Discrete to Continuous

#### ResNet Residual Connections
Standard ResNet block:
```
h_{t+1} = h_t + f(h_t, θ_t)
```

#### Limit as Depth → ∞
As step size Δt → 0:
```
dh(t)/dt = f(h(t), t, θ)
```

This transforms discrete layers into a continuous dynamical system.

#### Initial Value Problem
```
h(t_0) = h_0  (initial condition)
h(t_1) = h_0 + ∫_{t_0}^{t_1} f(h(t), t, θ) dt
```

The output is the solution at time t_1.

### 2. Neural ODE Formulation

#### Forward Pass
```
z(t_1) = z(t_0) + ∫_{t_0}^{t_1} f_θ(z(t), t) dt
```

Solved using ODE solvers:
- **Euler**: `z_{n+1} = z_n + h f(z_n, t_n)`
- **RK4**: Fourth-order Runge-Kutta
- **Adaptive**: DOPRI5, Fehlberg, etc.

#### Loss Function
```
L = loss(z(t_1), y_target)
```

### 3. Adjoint Sensitivity Method

#### The Backpropagation Challenge
Standard backprop requires storing all intermediate states.
For ODEs: infinitely many "layers" → infinite memory.

#### Adjoint State
Define adjoint state `a(t) = ∂L/∂z(t)`:
```
da(t)/dt = -a(t)^T ∂f(z(t),t,θ)/∂z
```

#### Gradients w.r.t. Parameters
```
∂L/∂θ = -∫_{t_1}^{t_0} a(t)^T ∂f(z(t),t,θ)/∂θ dt
```

#### Gradient w.r.t. Initial State
```
∂L/∂z(t_0) = a(t_0)
```

#### Augmented System
Solve jointly:
```
d/dt [z(t), a(t), ∂L/∂θ] = [f(z,t,θ), -a^T ∂f/∂z, -a^T ∂f/∂θ]
```

With terminal conditions:
```
z(t_1) = ODESolve(z(t_0), [t_0, t_1])
a(t_1) = ∂L/∂z(t_1)
∂L/∂θ(t_1) = 0
```

### 4. ODE Solvers

#### Fixed Step Methods

**Euler Method**:
```
z_{n+1} = z_n + h f(z_n, t_n)
Error: O(h²)
```

**RK4 Method**:
```
k1 = h f(z_n, t_n)
k2 = h f(z_n + k1/2, t_n + h/2)
k3 = h f(z_n + k2/2, t_n + h/2)
k4 = h f(z_n + k3, t_n + h)
z_{n+1} = z_n + (k1 + 2k2 + 2k3 + k4)/6
Error: O(h⁵)
```

#### Adaptive Methods

**Dormand-Prince (DOPRI5)**:
- 5th order method with 4th order error estimate
- Adaptive step size control
- Popular for Neural ODEs

**Error Control**:
```
error = ||z_5th - z_4th||
if error < tolerance:
    accept step, increase h
else:
    reject step, decrease h
```

### 5. Neural ODE Variants

#### Augmented Neural ODEs
Add extra dimensions to increase expressiveness:
```
z_aug = [z, a_1, a_2, ..., a_p]
dz_aug/dt = f_θ(z_aug, t)
```

#### Second-Order Neural ODEs
```
d²z/dt² = f_θ(z, dz/dt, t)
```

Equivalent first-order system:
```
[dz/dt, dv/dt] = [v, f_θ(z, v, t)]
```

#### Hamiltonian Neural Networks
Preserve energy in physical systems:
```
H(z) = T(p) + V(q)  # Hamiltonian
dq/dt = ∂H/∂p,  dp/dt = -∂H/∂q
```

### 6. Continuous Normalizing Flows

#### Change of Variables Formula
For invertible transformation T: x → z
```
log p_Z(z) = log p_X(x) - log |det(∂z/∂x)|
```

#### Instantaneous Change of Variables
For ODE: dz/dt = f(z, t)
```
d log p(z)/dt = -tr(∂f/∂z)
```

#### Neural ODE Flow
```
z(t_1) = z(t_0) + ∫_{t_0}^{t_1} f_θ(z(t), t) dt
log p(z(t_1)) = log p(z(t_0)) - ∫_{t_0}^{t_1} tr(∂f_θ/∂z) dt
```

#### FFJORD (Free-form Jacobian)
Approximate trace using Hutchinson estimator:
```
tr(∂f/∂z) ≈ E[ε^T (∂f/∂z) ε]
```
where ε ~ N(0, I).

### 7. Training Neural ODEs

#### Forward Pass Algorithm
```python
def forward(z0, t0, t1, theta):
    def ode_func(t, z):
        return f(z, t, theta)
    
    return ode_solve(ode_func, z0, [t0, t1])
```

#### Backward Pass Algorithm
```python  
def backward(dL_dz1, z0, z1, t0, t1, theta):
    # Augmented system
    def augmented_dynamics(t, augmented_state):
        z, a, dL_dtheta = augmented_state
        
        # Compute Jacobians
        dz_dt = f(z, t, theta)
        da_dt = -vjp(lambda z: f(z, t, theta), z, a)
        dL_dtheta_dt = -vjp(lambda theta: f(z, t, theta), theta, a)
        
        return [dz_dt, da_dt, dL_dtheta_dt]
    
    # Initial conditions (at t1, going backward)
    aug_init = [z1, dL_dz1, zeros_like(theta)]
    
    # Solve backward
    aug_final = ode_solve(augmented_dynamics, aug_init, [t1, t0])
    
    return aug_final[1], aug_final[2]  # dL_dz0, dL_dtheta
```

### 8. Applications

#### Continuous-Depth Models
```
class NeuralODELayer:
    def forward(self, x):
        return odeint(self.odefunc, x, self.t)
```

#### Time Series Modeling
**Latent ODEs**: Model irregular time series
```
z(t) ~ ODE(f_θ(z, t))
x(t_i) ~ p(x | g(z(t_i)))
```

#### Generative Models
**Continuous Normalizing Flows**:
```
p_X(x) = p_Z(z) |det(∂z/∂x)|
where z = NeuralODE(x)
```

#### Scientific Computing
**Physics-Informed Neural ODEs**:
```
Loss = MSE(prediction, data) + λ Physics_Loss
```

### 9. Computational Considerations

#### Memory Efficiency
**Advantage**: O(1) memory (only store current state)
**Disadvantage**: Recomputation during backward pass

#### Speed Trade-offs
- Adaptive solvers: Higher accuracy, variable cost
- Fixed-step solvers: Predictable cost, potential accuracy issues
- Solver tolerance affects speed/accuracy trade-off

#### Numerical Stability
- Stiff ODEs require implicit solvers
- Gradient clipping often necessary
- Initial condition sensitivity

### 10. Advanced Topics

#### Stochastic Differential Equations
```
dz = f(z, t)dt + g(z, t)dW
```
where dW is Brownian motion.

#### Neural Controlled Differential Equations
```
dz(t) = f_θ(z(t)) dX(t)
```
where X(t) is the control signal.

#### Meta-Learning with ODEs
Learn optimization dynamics:
```
dθ/dt = f_φ(θ, ∇L(θ))
```

## Implementation Details

See `exercise.py` for implementations of:
1. Basic Neural ODE with adjoint sensitivity
2. Various ODE solvers (Euler, RK4, DOPRI5)
3. Continuous normalizing flows
4. Augmented Neural ODEs
5. Training procedures and gradient computation
6. Time series modeling with latent ODEs
7. Computational efficiency optimizations
8. Numerical stability analysis