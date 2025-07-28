"""
Wasserstein GAN Implementation Exercise

Implement Wasserstein GAN with Earth Mover's distance and Kantorovich-Rubinstein duality.
Focus on understanding the theoretical improvements over vanilla GANs and practical training.

Key concepts:
- Earth Mover's (Wasserstein) distance
- Kantorovich-Rubinstein duality
- Lipschitz constraint and weight clipping
- Critic network instead of discriminator
- Improved training stability

Author: ML-from-Scratch Course
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class WassersteinCritic(nn.Module):
    """
    Critic network for Wasserstein GAN.
    
    TODO: Implement critic that estimates Wasserstein distance.
    
    Key differences from discriminator:
    - No sigmoid activation (outputs real values)
    - Lipschitz constraint enforcement
    - Estimates f(x) in Kantorovich-Rubinstein duality
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256):
        """
        Initialize critic network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
        """
        super(WassersteinCritic, self).__init__()
        
        # TODO: Define critic architecture
        # Note: No batch normalization in critic (affects Lipschitz constraint)
        # Note: No sigmoid at the end (outputs real values)
        self.model = None
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Forward pass through critic.
        
        Args:
            x: Input samples (batch_size, input_dim)
            
        Returns:
            Critic scores (batch_size, 1) - real values, not probabilities
        """
        pass
    
    def enforce_lipschitz_constraint(self, clip_value: float = 0.01):
        """
        TODO: Enforce Lipschitz constraint via weight clipping.
        
        Clips all weights to [-clip_value, clip_value] to approximately
        enforce 1-Lipschitz constraint.
        
        Args:
            clip_value: Clipping threshold
        """
        pass


class WassersteinGenerator(nn.Module):
    """
    Generator for Wasserstein GAN.
    
    TODO: Implement generator (similar to vanilla GAN but trained differently).
    """
    
    def __init__(self, noise_dim: int = 100, hidden_dim: int = 256, 
                 output_dim: int = 784):
        """
        Initialize generator.
        
        Args:
            noise_dim: Dimension of input noise
            hidden_dim: Hidden layer dimension  
            output_dim: Output dimension
        """
        super(WassersteinGenerator, self).__init__()
        self.noise_dim = noise_dim
        
        # TODO: Define generator architecture
        self.model = None
        pass
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        TODO: Forward pass through generator.
        
        Args:
            z: Noise tensor (batch_size, noise_dim)
            
        Returns:
            Generated samples (batch_size, output_dim)
        """
        pass


class WassersteinGAN:
    """
    Wasserstein GAN implementation.
    
    TODO: Implement WGAN training with Earth Mover's distance.
    """
    
    def __init__(self, generator: WassersteinGenerator, critic: WassersteinCritic,
                 noise_dim: int = 100, device: str = 'cpu'):
        """
        Initialize Wasserstein GAN.
        
        Args:
            generator: Generator network
            critic: Critic network
            noise_dim: Noise dimension
            device: Device for computation
        """
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.noise_dim = noise_dim
        self.device = device
        
        # Training history
        self.history = {
            'generator_loss': [],
            'critic_loss': [],
            'wasserstein_distance': [],
            'lipschitz_penalty': []
        }
    
    def compute_wasserstein_distance(self, real_data: torch.Tensor, 
                                   fake_data: torch.Tensor) -> torch.Tensor:
        """
        TODO: Compute Wasserstein distance estimate using critic.
        
        W(P_r, P_g) ≈ E[f(x)] - E[f(G(z))]
        where f is the optimal 1-Lipschitz function (critic)
        
        Args:
            real_data: Real data samples
            fake_data: Generated samples
            
        Returns:
            Wasserstein distance estimate
        """
        pass
    
    def train_step(self, real_data: torch.Tensor, 
                  g_optimizer: optim.Optimizer, c_optimizer: optim.Optimizer,
                  n_critic: int = 5, clip_value: float = 0.01) -> Dict[str, float]:
        """
        TODO: Implement WGAN training step.
        
        WGAN training procedure:
        1. Train critic n_critic times:
           - Maximize E[f(x)] - E[f(G(z))]
           - Clip weights to enforce Lipschitz constraint
        2. Train generator once:
           - Maximize E[f(G(z))]
        
        Args:
            real_data: Batch of real data
            g_optimizer: Generator optimizer
            c_optimizer: Critic optimizer  
            n_critic: Number of critic updates per generator update
            clip_value: Weight clipping value
            
        Returns:
            Dictionary with loss values and metrics
        """
        batch_size = real_data.size(0)
        
        # TODO: Train Critic
        for _ in range(n_critic):
            # TODO: Zero gradients
            # TODO: Compute critic loss: -[E[f(x)] - E[f(G(z))]]
            # TODO: Backward and optimize
            # TODO: Clip critic weights
            pass
        
        # TODO: Train Generator  
        # TODO: Maximize E[f(G(z))] = minimize -E[f(G(z))]
        
        # TODO: Compute metrics
        with torch.no_grad():
            wasserstein_dist = self.compute_wasserstein_distance(real_data, fake_data)
        
        return {
            'g_loss': 0.0,  # TODO: actual generator loss
            'c_loss': 0.0,  # TODO: actual critic loss
            'wasserstein_distance': wasserstein_dist.item() if wasserstein_dist else 0.0
        }
    
    def train(self, dataloader: DataLoader, epochs: int = 100,
              g_lr: float = 0.00005, c_lr: float = 0.00005,
              n_critic: int = 5, clip_value: float = 0.01) -> None:
        """
        TODO: Implement full WGAN training loop.
        
        Args:
            dataloader: DataLoader for training data
            epochs: Number of training epochs
            g_lr: Generator learning rate (typically lower than vanilla GAN)
            c_lr: Critic learning rate
            n_critic: Critic updates per generator update
            clip_value: Weight clipping value
        """
        # TODO: Initialize optimizers (RMSprop recommended for WGAN)
        g_optimizer = None
        c_optimizer = None
        
        self.generator.train()
        self.critic.train()
        
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_c_loss = 0.0
            epoch_wd = 0.0
            
            for batch_idx, (real_data, _) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                real_data = real_data.view(real_data.size(0), -1)  # Flatten
                
                # TODO: Perform training step
                metrics = self.train_step(real_data, g_optimizer, c_optimizer, 
                                        n_critic, clip_value)
                
                # Accumulate metrics
                epoch_g_loss += metrics['g_loss']
                epoch_c_loss += metrics['c_loss']
                epoch_wd += metrics['wasserstein_distance']
            
            # Average metrics
            num_batches = len(dataloader)
            epoch_g_loss /= num_batches
            epoch_c_loss /= num_batches
            epoch_wd /= num_batches
            
            # Store history
            self.history['generator_loss'].append(epoch_g_loss)
            self.history['critic_loss'].append(epoch_c_loss)
            self.history['wasserstein_distance'].append(epoch_wd)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] "
                      f"G_Loss: {epoch_g_loss:.4f} "
                      f"C_Loss: {epoch_c_loss:.4f} "
                      f"W_Distance: {epoch_wd:.4f}")
    
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """
        TODO: Generate samples using trained generator.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        self.generator.eval()
        with torch.no_grad():
            # TODO: Generate noise and pass through generator
            noise = None
            generated = None
            pass
        
        return generated
    
    def plot_training_history(self) -> None:
        """TODO: Plot WGAN training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Generator and Critic Loss
        axes[0, 0].plot(self.history['generator_loss'], label='Generator Loss')
        axes[0, 0].plot(self.history['critic_loss'], label='Critic Loss')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Wasserstein Distance
        axes[0, 1].plot(self.history['wasserstein_distance'], color='green')
        axes[0, 1].set_title('Wasserstein Distance')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Distance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss convergence analysis
        if len(self.history['generator_loss']) > 100:
            recent_g = np.mean(self.history['generator_loss'][-50:])
            recent_c = np.mean(self.history['critic_loss'][-50:])
            axes[1, 0].axhline(recent_g, color='blue', label=f'Recent G Loss: {recent_g:.3f}')
            axes[1, 0].axhline(recent_c, color='orange', label=f'Recent C Loss: {recent_c:.3f}')
            axes[1, 0].plot(self.history['generator_loss'][-100:], alpha=0.7)
            axes[1, 0].plot(self.history['critic_loss'][-100:], alpha=0.7)
            axes[1, 0].set_title('Recent Training Stability')
            axes[1, 0].legend()
        
        # Training balance
        if len(self.history['wasserstein_distance']) > 1:
            wd_gradient = np.gradient(self.history['wasserstein_distance'])
            axes[1, 1].plot(wd_gradient, color='red')
            axes[1, 1].set_title('Wasserstein Distance Gradient')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Change in Distance')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class WassersteinGANGP(WassersteinGAN):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP).
    
    TODO: Implement improved WGAN with gradient penalty instead of weight clipping.
    """
    
    def __init__(self, generator: WassersteinGenerator, critic: WassersteinCritic,
                 noise_dim: int = 100, device: str = 'cpu'):
        super().__init__(generator, critic, noise_dim, device)
        self.history['gradient_penalty'] = []
    
    def gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor,
                        lambda_gp: float = 10.0) -> torch.Tensor:
        """
        TODO: Compute gradient penalty for WGAN-GP.
        
        GP = λ * E[(||∇f(x̂)||₂ - 1)²]
        where x̂ = εx + (1-ε)G(z), ε ~ U[0,1]
        
        Args:
            real_data: Real data samples
            fake_data: Generated samples
            lambda_gp: Gradient penalty coefficient
            
        Returns:
            Gradient penalty loss
        """
        pass
    
    def train_step_gp(self, real_data: torch.Tensor,
                     g_optimizer: optim.Optimizer, c_optimizer: optim.Optimizer,
                     n_critic: int = 5, lambda_gp: float = 10.0) -> Dict[str, float]:
        """
        TODO: Implement WGAN-GP training step.
        
        Critic loss: -E[f(x)] + E[f(G(z))] + λ*GP
        Generator loss: -E[f(G(z))]
        
        Args:
            real_data: Real data batch
            g_optimizer: Generator optimizer
            c_optimizer: Critic optimizer
            n_critic: Critic updates per generator update
            lambda_gp: Gradient penalty coefficient
            
        Returns:
            Training metrics
        """
        pass


def compare_wgan_variants(real_data: np.ndarray) -> Dict:
    """
    TODO: Compare vanilla GAN, WGAN, and WGAN-GP.
    
    Args:
        real_data: Real dataset for comparison
        
    Returns:
        Comparison results
    """
    pass


def analyze_lipschitz_constraint(critic: WassersteinCritic, 
                               test_data: torch.Tensor) -> Dict[str, float]:
    """
    TODO: Analyze how well Lipschitz constraint is satisfied.
    
    Compute empirical Lipschitz constant:
    L ≈ max |f(x₁) - f(x₂)| / ||x₁ - x₂||
    
    Args:
        critic: Trained critic network
        test_data: Test data for analysis
        
    Returns:
        Lipschitz analysis results
    """
    pass


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_implement_wgan():
    """
    Exercise 1: Implement basic WGAN components.
    
    Tasks:
    1. Complete WassersteinCritic with weight clipping
    2. Complete WassersteinGenerator
    3. Test forward passes and weight clipping
    4. Verify architectural differences from vanilla GAN
    """
    print("Exercise 1: Implementing WGAN Components")
    print("=" * 50)
    
    # TODO: Create generator and critic
    generator = WassersteinGenerator(noise_dim=100, hidden_dim=256, output_dim=784)
    critic = WassersteinCritic(input_dim=784, hidden_dim=256)
    
    # TODO: Test with dummy data
    batch_size = 32
    noise = torch.randn(batch_size, 100)
    fake_data = torch.randn(batch_size, 784)
    
    print("Testing Generator:")
    try:
        gen_output = generator(noise)
        print(f"Generator output shape: {gen_output.shape}")
        print(f"Generator output range: [{gen_output.min():.3f}, {gen_output.max():.3f}]")
    except Exception as e:
        print(f"Generator error: {e}")
    
    print("\nTesting Critic:")
    try:
        critic_output = critic(fake_data)
        print(f"Critic output shape: {critic_output.shape}")
        print(f"Critic output range: [{critic_output.min():.3f}, {critic_output.max():.3f}]")
        print("Note: Critic outputs real values, not probabilities")
    except Exception as e:
        print(f"Critic error: {e}")
    
    print("\nTesting weight clipping:")
    try:
        # Get weight before clipping
        initial_weights = list(critic.parameters())[0].clone()
        
        # Apply clipping
        critic.enforce_lipschitz_constraint(clip_value=0.01)
        
        # Check weights after clipping
        clipped_weights = list(critic.parameters())[0]
        print(f"Weight range before clipping: [{initial_weights.min():.3f}, {initial_weights.max():.3f}]")
        print(f"Weight range after clipping: [{clipped_weights.min():.3f}, {clipped_weights.max():.3f}]")
    except Exception as e:
        print(f"Weight clipping error: {e}")


def exercise_2_wgan_training():
    """
    Exercise 2: Implement and test WGAN training.
    
    Tasks:
    1. Complete WGAN training loop
    2. Test on synthetic 2D data
    3. Compare stability with vanilla GAN
    4. Analyze Wasserstein distance evolution
    """
    print("\nExercise 2: WGAN Training")
    print("=" * 50)
    
    # Create synthetic 2D data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Mixture of Gaussians
    n_samples = 1000
    data = []
    data.append(np.random.normal([2, 2], 0.5, (n_samples//2, 2)))
    data.append(np.random.normal([-2, -2], 0.5, (n_samples//2, 2)))
    
    synthetic_data = np.vstack(data)
    
    # Convert to PyTorch
    dataset = TensorDataset(torch.FloatTensor(synthetic_data), torch.zeros(len(synthetic_data)))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # TODO: Create and train WGAN
    generator = WassersteinGenerator(noise_dim=10, hidden_dim=32, output_dim=2)
    critic = WassersteinCritic(input_dim=2, hidden_dim=32)
    
    wgan = WassersteinGAN(generator, critic, noise_dim=10)
    
    print("Starting WGAN training...")
    wgan.train(dataloader, epochs=100, g_lr=0.00005, c_lr=0.00005, n_critic=5)
    
    # TODO: Plot results
    wgan.plot_training_history()
    
    # TODO: Generate and visualize samples
    generated_samples = wgan.generate_samples(500)
    
    if generated_samples is not None:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], alpha=0.6, label='Real Data')
        plt.title('Real Data Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        gen_np = generated_samples.detach().cpu().numpy()
        plt.scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.6, label='Generated Data', color='red')
        plt.title('WGAN Generated Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


def exercise_3_wgan_gp():
    """
    Exercise 3: Implement WGAN with Gradient Penalty.
    
    Tasks:
    1. Implement gradient penalty computation
    2. Complete WGAN-GP training
    3. Compare WGAN vs WGAN-GP
    4. Analyze gradient penalty behavior
    """
    print("\nExercise 3: WGAN with Gradient Penalty")
    print("=" * 50)
    
    # TODO: Test gradient penalty computation
    generator = WassersteinGenerator(noise_dim=10, hidden_dim=32, output_dim=2)
    critic = WassersteinCritic(input_dim=2, hidden_dim=32)
    
    wgan_gp = WassersteinGANGP(generator, critic, noise_dim=10)
    
    # Test gradient penalty with dummy data
    real_data = torch.randn(32, 2)
    fake_data = torch.randn(32, 2)
    
    try:
        gp = wgan_gp.gradient_penalty(real_data, fake_data, lambda_gp=10.0)
        print(f"Gradient penalty computed: {gp.item():.4f}")
    except Exception as e:
        print(f"Gradient penalty error: {e}")
        print("TODO: Implement gradient penalty computation")
    
    # TODO: Compare training with and without gradient penalty
    print("\nTODO: Implement WGAN-GP training and comparison")


def exercise_4_lipschitz_analysis():
    """
    Exercise 4: Analyze Lipschitz constraint satisfaction.
    
    Tasks:
    1. Measure empirical Lipschitz constant
    2. Compare weight clipping vs gradient penalty
    3. Study effect on training stability
    4. Visualize critic function smoothness
    """
    print("\nExercise 4: Lipschitz Constraint Analysis")
    print("=" * 50)
    
    # TODO: Create test critic
    critic = WassersteinCritic(input_dim=2, hidden_dim=32)
    
    # Generate test data
    test_data = torch.randn(100, 2)
    
    print("Analyzing Lipschitz constraint satisfaction:")
    
    # TODO: Before weight clipping
    lipschitz_before = analyze_lipschitz_constraint(critic, test_data)
    print("Before weight clipping: TODO")
    
    # TODO: After weight clipping
    critic.enforce_lipschitz_constraint(clip_value=0.01)
    lipschitz_after = analyze_lipschitz_constraint(critic, test_data)
    print("After weight clipping: TODO")
    
    print("\nTODO: Implement Lipschitz analysis tools")


def exercise_5_wgan_theory():
    """
    Exercise 5: Study Wasserstein distance theory.
    
    Tasks:
    1. Understand Earth Mover's distance
    2. Verify Kantorovich-Rubinstein duality
    3. Compare with JS divergence
    4. Analyze convergence properties
    """
    print("\nExercise 5: Wasserstein Distance Theory")
    print("=" * 50)
    
    # TODO: Create 1D distributions for analysis
    x_points = np.linspace(-4, 4, 1000)
    
    # Data distribution
    p_data = np.exp(-0.5 * (x_points - 1)**2) / np.sqrt(2 * np.pi)
    
    # Generator distributions at different training stages
    generators = [
        np.exp(-0.5 * (x_points + 2)**2) / np.sqrt(2 * np.pi),  # Far from data
        np.exp(-0.5 * (x_points)**2) / np.sqrt(2 * np.pi),      # Closer
        np.exp(-0.5 * (x_points - 1)**2) / np.sqrt(2 * np.pi)   # Match
    ]
    
    generator_names = ["Initial", "Intermediate", "Converged"]
    
    print("Theoretical Analysis:")
    print("1. Wasserstein distance has nice convergence properties")
    print("2. Provides meaningful gradients even when supports don't overlap")
    print("3. Continuous and differentiable under mild conditions")
    print("4. Does not saturate like JS divergence")
    
    # TODO: Visualize distributions and distances
    plt.figure(figsize=(15, 5))
    
    for i, (p_gen, name) in enumerate(zip(generators, generator_names)):
        plt.subplot(1, 3, i+1)
        plt.plot(x_points, p_data, label='Data Distribution', linewidth=2)
        plt.plot(x_points, p_gen, label='Generator Distribution', linewidth=2)
        plt.title(f'{name} Generator')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # TODO: Compute and display Wasserstein distance
        # wd = compute_wasserstein_1d(p_data, p_gen, x_points)
        # plt.text(0.05, 0.95, f'W₁ ≈ {wd:.3f}', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
    
    print("\nKey advantages of Wasserstein distance:")
    print("- Provides meaningful gradients everywhere")
    print("- No vanishing gradient problem")
    print("- Correlates with sample quality")
    print("- Stable training dynamics")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Wasserstein GAN Implementation Exercises")
    print("=" * 60)
    
    # Run exercises
    exercise_1_implement_wgan()
    exercise_2_wgan_training()
    exercise_3_wgan_gp()
    exercise_4_lipschitz_analysis()
    exercise_5_wgan_theory()
    
    print("\n" + "=" * 60)
    print("COMPLETION CHECKLIST:")
    print("✓ Implement Wasserstein Critic with weight clipping")
    print("✓ Implement WGAN training loop")
    print("✓ Implement WGAN-GP with gradient penalty")
    print("✓ Analyze Lipschitz constraint satisfaction")
    print("✓ Study Wasserstein distance theory")
    
    print("\nKey insights from WGAN:")
    print("- Earth Mover's distance provides better training signals")
    print("- Lipschitz constraint is crucial for valid distance")
    print("- Weight clipping vs gradient penalty trade-offs")
    print("- Improved stability over vanilla GANs")
    print("- Wasserstein distance correlates with sample quality") 