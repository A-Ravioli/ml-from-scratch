"""
Planar Flows Implementation Exercise

Implement planar normalizing flows for learning complex probability distributions.
Focus on invertible transformations and the change of variables formula.

Key concepts:
- Normalizing flows and invertible transformations
- Change of variables formula for probability densities
- Planar flow transformations
- Jacobian determinant computation
- Flow composition and expressive power

Author: ML-from-Scratch Course
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, Uniform
import math


class PlanarFlow(nn.Module):
    """
    Planar flow transformation.
    
    TODO: Implement planar flow layer.
    
    Transformation: f(z) = z + u * h(w^T z + b)
    where h is a nonlinear activation (typically tanh).
    
    For invertibility, need to ensure det(∂f/∂z) > 0.
    """
    
    def __init__(self, dim: int, activation: str = 'tanh'):
        """
        Initialize planar flow.
        
        Args:
            dim: Dimensionality of the input
            activation: Activation function ('tanh' or 'relu')
        """
        super(PlanarFlow, self).__init__()
        self.dim = dim
        
        # TODO: Initialize learnable parameters
        # u: direction vector (dim,)
        # w: weight vector (dim,)
        # b: bias scalar
        self.u = None
        self.w = None
        self.b = None
        
        # TODO: Choose activation function
        if activation == 'tanh':
            self.h = torch.tanh
            self.h_prime = lambda x: 1 - torch.tanh(x)**2
        elif activation == 'relu':
            self.h = torch.relu
            self.h_prime = lambda x: (x > 0).float()
        else:
            raise ValueError("Activation must be 'tanh' or 'relu'")
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Forward transformation and log determinant.
        
        Args:
            z: Input tensor (batch_size, dim)
            
        Returns:
            z_new: Transformed tensor (batch_size, dim)
            log_det_jacobian: Log determinant of Jacobian (batch_size,)
        """
        pass
    
    def ensure_invertibility(self):
        """
        TODO: Ensure the transformation is invertible.
        
        For planar flows to be invertible, we need:
        u^T w ≥ -1
        
        If not satisfied, we can constrain u using:
        û = u + (m(w^T u) - w^T u) * w / ||w||²
        where m(x) = -1 + log(1 + exp(x))
        """
        pass


class NormalizingFlow(nn.Module):
    """
    Composition of normalizing flow transformations.
    
    TODO: Implement full normalizing flow model.
    """
    
    def __init__(self, dim: int, n_flows: int, flow_type: str = 'planar'):
        """
        Initialize normalizing flow.
        
        Args:
            dim: Dimensionality
            n_flows: Number of flow layers
            flow_type: Type of flow ('planar', etc.)
        """
        super(NormalizingFlow, self).__init__()
        self.dim = dim
        self.n_flows = n_flows
        
        # TODO: Create flow layers
        if flow_type == 'planar':
            self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])
        else:
            raise ValueError("Only 'planar' flows implemented")
        
        # Base distribution (standard Gaussian)
        self.base_dist = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Apply sequence of flow transformations.
        
        Args:
            z: Base samples (batch_size, dim)
            
        Returns:
            x: Transformed samples (batch_size, dim)
            log_det_jacobian: Total log determinant (batch_size,)
        """
        pass
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        TODO: Sample from the learned distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Samples from the transformed distribution
        """
        pass
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Compute log probability of data.
        
        Uses change of variables formula:
        log p_X(x) = log p_Z(z) - log |det(∂f/∂z)|
        
        Args:
            x: Data points (batch_size, dim)
            
        Returns:
            Log probabilities (batch_size,)
        """
        pass
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Compute inverse transformation (if possible).
        
        For planar flows, this requires iterative methods.
        
        Args:
            x: Transformed samples
            
        Returns:
            z: Base space samples
            log_det_jacobian: Log determinant
        """
        pass


def train_normalizing_flow(flow: NormalizingFlow, data: torch.Tensor,
                          epochs: int = 1000, lr: float = 0.001) -> List[float]:
    """
    TODO: Train normalizing flow to match target distribution.
    
    Args:
        flow: Normalizing flow model
        data: Target data samples
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Training losses
    """
    pass


def evaluate_flow_quality(flow: NormalizingFlow, test_data: torch.Tensor) -> Dict[str, float]:
    """
    TODO: Evaluate quality of learned distribution.
    
    Metrics:
    - Log likelihood on test data
    - KL divergence estimate
    - Sample quality metrics
    
    Args:
        flow: Trained flow
        test_data: Test data
        
    Returns:
        Evaluation metrics
    """
    pass


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_planar_flow_basics():
    """
    Exercise 1: Implement and test basic planar flow.
    
    Tasks:
    1. Complete PlanarFlow forward pass
    2. Implement Jacobian determinant computation
    3. Test invertibility constraints
    4. Visualize transformations
    """
    print("Exercise 1: Planar Flow Basics")
    print("=" * 50)
    
    # TODO: Create planar flow
    dim = 2
    flow = PlanarFlow(dim)
    
    print(f"Created planar flow for {dim}D data")
    
    # TODO: Test forward pass
    batch_size = 100
    z = torch.randn(batch_size, dim)
    
    try:
        z_new, log_det = flow(z)
        print(f"Forward pass successful:")
        print(f"  Input shape: {z.shape}")
        print(f"  Output shape: {z_new.shape}")
        print(f"  Log det shape: {log_det.shape}")
        print(f"  Log det range: [{log_det.min():.3f}, {log_det.max():.3f}]")
    except Exception as e:
        print(f"Forward pass error: {e}")
        print("TODO: Implement planar flow forward pass")
    
    # TODO: Test invertibility constraint
    try:
        flow.ensure_invertibility()
        print("Invertibility constraint applied")
    except Exception as e:
        print(f"Invertibility error: {e}")
        print("TODO: Implement invertibility constraint")
    
    # TODO: Visualize transformation
    if dim == 2:
        print("\nVisualizing 2D transformation:")
        # Create grid of points
        x = torch.linspace(-3, 3, 20)
        y = torch.linspace(-3, 3, 20)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        try:
            transformed_points, _ = flow(grid_points)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.scatter(grid_points[:, 0], grid_points[:, 1], alpha=0.6, s=20)
            plt.title('Original Grid')
            plt.xlabel('z₁')
            plt.ylabel('z₂')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            plt.subplot(1, 2, 2)
            transformed_np = transformed_points.detach().numpy()
            plt.scatter(transformed_np[:, 0], transformed_np[:, 1], alpha=0.6, s=20, color='red')
            plt.title('Transformed Grid')
            plt.xlabel('x₁')
            plt.ylabel('x₂')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization error: {e}")


def exercise_2_normalizing_flow_model():
    """
    Exercise 2: Implement full normalizing flow model.
    
    Tasks:
    1. Complete NormalizingFlow class
    2. Implement flow composition
    3. Test sampling and log probability
    4. Study effect of number of layers
    """
    print("\nExercise 2: Normalizing Flow Model")
    print("=" * 50)
    
    # TODO: Create normalizing flow with multiple layers
    dim = 2
    n_flows_list = [1, 5, 10]
    
    for n_flows in n_flows_list:
        print(f"\nTesting flow with {n_flows} layers:")
        
        flow = NormalizingFlow(dim, n_flows)
        
        # TODO: Test sampling
        try:
            samples = flow.sample(100)
            print(f"  Sampling successful: {samples.shape}")
            print(f"  Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
        except Exception as e:
            print(f"  Sampling error: {e}")
        
        # TODO: Test log probability
        try:
            test_data = torch.randn(50, dim)
            log_probs = flow.log_prob(test_data)
            print(f"  Log prob successful: {log_probs.shape}")
            print(f"  Log prob range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")
        except Exception as e:
            print(f"  Log prob error: {e}")
    
    print("\nTODO: Implement NormalizingFlow methods")


def exercise_3_train_on_synthetic_data():
    """
    Exercise 3: Train flow on synthetic 2D distributions.
    
    Tasks:
    1. Create target distributions (moons, circles, etc.)
    2. Train normalizing flow to match
    3. Visualize learned distribution
    4. Compare different architectures
    """
    print("\nExercise 3: Training on Synthetic Data")
    print("=" * 50)
    
    # TODO: Create target distributions
    torch.manual_seed(42)
    n_samples = 1000
    
    # Mixture of Gaussians
    mixture_data = torch.cat([
        torch.randn(n_samples//2, 2) * 0.5 + torch.tensor([2., 2.]),
        torch.randn(n_samples//2, 2) * 0.5 + torch.tensor([-2., -2.])
    ])
    
    print("Created mixture of Gaussians target distribution")
    
    # TODO: Train different flow architectures
    architectures = [
        (2, "Simple"),
        (5, "Medium"),
        (10, "Complex")
    ]
    
    trained_flows = {}
    
    for n_flows, name in architectures:
        print(f"\nTraining {name} flow ({n_flows} layers)...")
        
        flow = NormalizingFlow(2, n_flows)
        
        # TODO: Train flow
        try:
            losses = train_normalizing_flow(flow, mixture_data, epochs=500, lr=0.001)
            trained_flows[name] = (flow, losses)
            print(f"  Training completed. Final loss: TODO")
        except Exception as e:
            print(f"  Training error: {e}")
            print("  TODO: Implement training function")
    
    # TODO: Visualize results
    if trained_flows:
        n_flows = len(trained_flows)
        fig, axes = plt.subplots(2, n_flows, figsize=(4*n_flows, 8))
        
        for i, (name, (flow, losses)) in enumerate(trained_flows.items()):
            # Plot target vs learned distribution
            axes[0, i].scatter(mixture_data[:, 0], mixture_data[:, 1], 
                             alpha=0.6, s=20, label='Target')
            
            try:
                learned_samples = flow.sample(1000)
                learned_np = learned_samples.detach().numpy()
                axes[0, i].scatter(learned_np[:, 0], learned_np[:, 1], 
                                 alpha=0.6, s=20, color='red', label='Learned')
            except:
                pass
            
            axes[0, i].set_title(f'{name} Flow')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot training loss
            if losses:
                axes[1, i].plot(losses)
                axes[1, i].set_title(f'{name} Training Loss')
                axes[1, i].set_xlabel('Epoch')
                axes[1, i].set_ylabel('Negative Log Likelihood')
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def exercise_4_expressive_power():
    """
    Exercise 4: Study expressive power of planar flows.
    
    Tasks:
    1. Test on complex distributions
    2. Analyze effect of flow depth
    3. Compare with other generative models
    4. Study failure modes
    """
    print("\nExercise 4: Expressive Power Analysis")
    print("=" * 50)
    
    # TODO: Create challenging target distributions
    
    # Spiral distribution
    def create_spiral(n_samples=500):
        t = torch.linspace(0, 4*np.pi, n_samples)
        r = t / (4*np.pi)
        x = r * torch.cos(t) + 0.1 * torch.randn(n_samples)
        y = r * torch.sin(t) + 0.1 * torch.randn(n_samples)
        return torch.stack([x, y], dim=1)
    
    # Ring distribution  
    def create_ring(n_samples=500, radius=2.0):
        theta = torch.rand(n_samples) * 2 * np.pi
        r = radius + 0.3 * torch.randn(n_samples)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=1)
    
    target_distributions = {
        'Spiral': create_spiral(),
        'Ring': create_ring()
    }
    
    # TODO: Test different flow depths
    flow_depths = [2, 5, 10, 20]
    
    results = {}
    
    for dist_name, target_data in target_distributions.items():
        print(f"\nTesting on {dist_name} distribution:")
        
        results[dist_name] = {}
        
        for depth in flow_depths:
            print(f"  Flow depth {depth}...")
            
            flow = NormalizingFlow(2, depth)
            
            # TODO: Train and evaluate
            try:
                losses = train_normalizing_flow(flow, target_data, epochs=300, lr=0.001)
                
                # Evaluate quality
                metrics = evaluate_flow_quality(flow, target_data)
                
                results[dist_name][depth] = {
                    'flow': flow,
                    'losses': losses,
                    'metrics': metrics
                }
                
                print(f"    Final loss: TODO")
                print(f"    Quality metrics: TODO")
                
            except Exception as e:
                print(f"    Error: {e}")
    
    # TODO: Visualize expressive power analysis
    print("\nExpressive Power Summary:")
    print("- Deeper flows can capture more complex distributions")
    print("- Planar flows have limited expressive power per layer")
    print("- Some distributions may require many layers")
    print("- Trade-off between expressivity and computational cost")


def exercise_5_theoretical_analysis():
    """
    Exercise 5: Theoretical analysis of planar flows.
    
    Tasks:
    1. Analyze transformation properties
    2. Study Jacobian determinant behavior
    3. Understand expressive limitations
    4. Connection to universal approximation
    """
    print("\nExercise 5: Theoretical Analysis")
    print("=" * 50)
    
    # TODO: Analyze single planar flow properties
    flow = PlanarFlow(2)
    
    print("Planar Flow Properties:")
    print("1. Transformation: f(z) = z + u * tanh(w^T z + b)")
    print("2. Adds rank-1 update in direction of u")
    print("3. Jacobian: I + u * tanh'(w^T z + b) * w^T")
    print("4. Log det: log(1 + u^T w * tanh'(w^T z + b))")
    
    # TODO: Visualize Jacobian determinant
    print("\nJacobian Determinant Analysis:")
    
    # Create test points
    z_test = torch.randn(1000, 2)
    
    try:
        _, log_det = flow(z_test)
        det_values = torch.exp(log_det)
        
        print(f"Determinant statistics:")
        print(f"  Mean: {det_values.mean():.3f}")
        print(f"  Std: {det_values.std():.3f}")
        print(f"  Min: {det_values.min():.3f}")
        print(f"  Max: {det_values.max():.3f}")
        
        # Plot determinant distribution
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(det_values.detach().numpy(), bins=50, alpha=0.7)
        plt.xlabel('Jacobian Determinant')
        plt.ylabel('Frequency')
        plt.title('Distribution of Jacobian Determinants')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(z_test[:, 0], z_test[:, 1], c=det_values.detach().numpy(), 
                   alpha=0.6, s=20, cmap='viridis')
        plt.colorbar(label='Jacobian Determinant')
        plt.xlabel('z₁')
        plt.ylabel('z₂')
        plt.title('Spatial Distribution of Determinants')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Analysis error: {e}")
    
    print("\nTheoretical Insights:")
    print("- Planar flows are volume-preserving on average")
    print("- Limited expressive power: only rank-1 updates")
    print("- Need many layers for complex distributions")
    print("- Connection to neural ODEs and continuous flows")
    print("- Trade-off: expressivity vs computational efficiency")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Planar Flows Implementation Exercises")
    print("=" * 60)
    
    # Run exercises
    exercise_1_planar_flow_basics()
    exercise_2_normalizing_flow_model()
    exercise_3_train_on_synthetic_data()
    exercise_4_expressive_power()
    exercise_5_theoretical_analysis()
    
    print("\n" + "=" * 60)
    print("COMPLETION CHECKLIST:")
    print("✓ Implement PlanarFlow transformation")
    print("✓ Implement Jacobian determinant computation")
    print("✓ Implement NormalizingFlow composition")
    print("✓ Implement training procedure")
    print("✓ Analyze expressive power and limitations")
    
    print("\nKey insights from Planar Flows:")
    print("- Change of variables enables flexible distributions")
    print("- Jacobian determinant is crucial for valid densities")
    print("- Invertibility constraints ensure well-defined flows")
    print("- Depth increases expressivity but adds computational cost")
    print("- Foundation for more advanced flow architectures") 