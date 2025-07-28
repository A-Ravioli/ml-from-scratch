"""
DDPM (Denoising Diffusion Probabilistic Models) Implementation Exercise

Implement the foundational diffusion model from "Denoising Diffusion Probabilistic Models"
by Ho et al. Focus on the forward diffusion process, reverse denoising, and training.

Key concepts:
- Forward diffusion process with Gaussian noise
- Reverse denoising process with learned neural network
- Variational lower bound and loss function
- Reparameterization trick and noise scheduling
- Connection to score-based models

Author: ML-from-Scratch Course
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Callable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math


class NoiseScheduler:
    """
    Noise scheduling for diffusion process.
    
    TODO: Implement different noise schedules (linear, cosine, etc.).
    """
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, 
                 beta_end: float = 0.02, schedule_type: str = 'linear'):
        """
        Initialize noise scheduler.
        
        Args:
            num_timesteps: Number of diffusion timesteps T
            beta_start: Starting noise level
            beta_end: Ending noise level
            schedule_type: Type of schedule ('linear', 'cosine')
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        # TODO: Compute noise schedules
        self.betas = None
        self.alphas = None
        self.alphas_cumprod = None
        self.alphas_cumprod_prev = None
        
        self._compute_schedule()
    
    def _compute_schedule(self):
        """
        TODO: Compute noise schedule parameters.
        
        Key quantities:
        - β_t: noise variance at step t
        - α_t = 1 - β_t
        - ᾱ_t = ∏(α_s) for s=1 to t
        """
        pass
    
    def get_variance(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        TODO: Get variance for timestep t.
        
        Args:
            t: Timestep tensor
            x_shape: Shape for broadcasting
            
        Returns:
            Variance tensor
        """
        pass
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        TODO: Sample from q(x_t | x_0) - forward diffusion.
        
        q(x_t | x_0) = N(√ᾱ_t x_0, (1 - ᾱ_t)I)
        
        Args:
            x_start: Original data x_0
            t: Timestep
            noise: Optional noise (if None, sample fresh)
            
        Returns:
            Noisy data x_t
        """
        pass
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, 
                                 t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Compute posterior q(x_{t-1} | x_t, x_0).
        
        This is used for computing the true reverse process.
        
        Args:
            x_start: Original data x_0
            x_t: Noisy data at step t
            t: Timestep
            
        Returns:
            Mean and variance of posterior
        """
        pass


class UNet(nn.Module):
    """
    U-Net architecture for denoising.
    
    TODO: Implement U-Net with time embedding for ε_θ(x_t, t).
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 time_emb_dim: int = 128, hidden_dims: List[int] = [64, 128, 256]):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels  
            time_emb_dim: Time embedding dimension
            hidden_dims: Hidden dimensions for each level
        """
        super(UNet, self).__init__()
        self.time_emb_dim = time_emb_dim
        
        # TODO: Time embedding layers
        self.time_mlp = None
        
        # TODO: Downsampling path
        self.downs = nn.ModuleList()
        
        # TODO: Upsampling path  
        self.ups = nn.ModuleList()
        
        # TODO: Output layer
        self.output = None
        
        pass
    
    def time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        TODO: Create sinusoidal time embeddings.
        
        Args:
            timesteps: Timestep tensor
            
        Returns:
            Time embeddings
        """
        pass
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        TODO: Forward pass through U-Net.
        
        Args:
            x: Noisy input x_t
            timesteps: Timestep t
            
        Returns:
            Predicted noise ε_θ(x_t, t)
        """
        pass


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.
    
    TODO: Implement complete DDPM with training and sampling.
    """
    
    def __init__(self, model: UNet, noise_scheduler: NoiseScheduler, 
                 device: str = 'cpu'):
        """
        Initialize DDPM.
        
        Args:
            model: Denoising model (U-Net)
            noise_scheduler: Noise scheduling
            device: Device for computation
        """
        super(DDPM, self).__init__()
        self.model = model.to(device)
        self.noise_scheduler = noise_scheduler
        self.device = device
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def compute_loss(self, x_start: torch.Tensor) -> torch.Tensor:
        """
        TODO: Compute DDPM training loss.
        
        Loss: E[||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²]
        
        Args:
            x_start: Original data batch
            
        Returns:
            Loss tensor
        """
        pass
    
    def p_sample_step(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Single step of reverse sampling p(x_{t-1} | x_t).
        
        Uses reparameterization: x_{t-1} = μ_θ(x_t, t) + σ_t z
        where μ_θ is computed from ε_θ.
        
        Args:
            x_t: Current noisy sample
            t: Current timestep
            
        Returns:
            Denoised sample x_{t-1}
        """
        pass
    
    def p_sample(self, shape: Tuple[int, ...], return_all_timesteps: bool = False) -> torch.Tensor:
        """
        TODO: Generate samples via reverse diffusion.
        
        Start from pure noise and iteratively denoise.
        
        Args:
            shape: Shape of samples to generate
            return_all_timesteps: Whether to return intermediate steps
            
        Returns:
            Generated samples (and optionally all timesteps)
        """
        pass
    
    def train_step(self, batch: torch.Tensor, optimizer: optim.Optimizer) -> float:
        """
        TODO: Single training step.
        
        Args:
            batch: Training batch
            optimizer: Optimizer
            
        Returns:
            Loss value
        """
        pass
    
    def train(self, dataloader: DataLoader, epochs: int = 100, 
              lr: float = 0.001, val_dataloader: Optional[DataLoader] = None) -> None:
        """
        TODO: Full training loop.
        
        Args:
            dataloader: Training data
            epochs: Number of epochs
            lr: Learning rate
            val_dataloader: Validation data
        """
        pass


class DDPMSampler:
    """
    Advanced sampling methods for DDPM.
    
    TODO: Implement different sampling strategies.
    """
    
    def __init__(self, ddpm: DDPM):
        self.ddpm = ddpm
    
    def ddim_sample(self, shape: Tuple[int, ...], eta: float = 0.0, 
                   steps: int = 50) -> torch.Tensor:
        """
        TODO: DDIM (deterministic) sampling.
        
        Faster sampling with fewer steps using deterministic process.
        
        Args:
            shape: Sample shape
            eta: Stochasticity parameter (0 = deterministic)
            steps: Number of sampling steps
            
        Returns:
            Generated samples
        """
        pass
    
    def ancestral_sample(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        TODO: Standard ancestral sampling (full T steps).
        
        Args:
            shape: Sample shape
            
        Returns:
            Generated samples
        """
        pass


def evaluate_ddpm_quality(ddpm: DDPM, test_data: torch.Tensor, 
                         n_samples: int = 1000) -> Dict[str, float]:
    """
    TODO: Evaluate DDPM sample quality.
    
    Metrics:
    - FID (Fréchet Inception Distance)
    - IS (Inception Score)  
    - Sample diversity
    - Reconstruction quality
    
    Args:
        ddpm: Trained DDPM model
        test_data: Test dataset
        n_samples: Number of samples for evaluation
        
    Returns:
        Quality metrics
    """
    pass


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_noise_scheduling():
    """
    Exercise 1: Implement and analyze noise scheduling.
    
    Tasks:
    1. Implement linear and cosine noise schedules
    2. Visualize forward diffusion process
    3. Study effect of different schedules
    4. Analyze noise addition over time
    """
    print("Exercise 1: Noise Scheduling")
    print("=" * 50)
    
    # TODO: Test different noise schedules
    schedules = {
        'Linear': NoiseScheduler(num_timesteps=1000, schedule_type='linear'),
        'Cosine': NoiseScheduler(num_timesteps=1000, schedule_type='cosine')
    }
    
    # TODO: Visualize schedules
    plt.figure(figsize=(15, 5))
    
    for i, (name, scheduler) in enumerate(schedules.items()):
        plt.subplot(1, 3, i+1)
        
        try:
            timesteps = torch.arange(scheduler.num_timesteps)
            betas = scheduler.betas
            alphas_cumprod = scheduler.alphas_cumprod
            
            plt.plot(timesteps, betas, label='β_t', alpha=0.7)
            plt.plot(timesteps, 1 - alphas_cumprod, label='1 - ᾱ_t', alpha=0.7)
            plt.title(f'{name} Schedule')
            plt.xlabel('Timestep t')
            plt.ylabel('Noise Level')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        except Exception as e:
            plt.text(0.5, 0.5, f'TODO: Implement\n{name} schedule', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{name} Schedule')
    
    # Test forward diffusion
    plt.subplot(1, 3, 3)
    try:
        # Create simple 2D data
        x_start = torch.randn(1, 2)
        scheduler = schedules['Linear']
        
        # Sample at different timesteps
        timesteps = [0, 100, 500, 999]
        for t in timesteps:
            t_tensor = torch.tensor([t])
            x_t = scheduler.q_sample(x_start, t_tensor)
            plt.scatter(x_t[0, 0], x_t[0, 1], label=f't={t}', s=100, alpha=0.7)
        
        plt.title('Forward Diffusion')
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    except Exception as e:
        plt.text(0.5, 0.5, 'TODO: Implement\nq_sample', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Forward Diffusion')
    
    plt.tight_layout()
    plt.show()
    
    print("TODO: Implement noise scheduling methods")


def exercise_2_unet_architecture():
    """
    Exercise 2: Implement U-Net denoising model.
    
    Tasks:
    1. Complete U-Net architecture with time embedding
    2. Test forward pass with different inputs
    3. Analyze parameter count and model capacity
    4. Visualize attention to time information
    """
    print("\nExercise 2: U-Net Architecture")
    print("=" * 50)
    
    # TODO: Create U-Net model
    model = UNet(in_channels=1, out_channels=1, time_emb_dim=128)
    
    print(f"Created U-Net model")
    
    # TODO: Test time embedding
    try:
        timesteps = torch.randint(0, 1000, (10,))
        time_emb = model.time_embedding(timesteps)
        print(f"Time embedding shape: {time_emb.shape}")
        print(f"Time embedding range: [{time_emb.min():.3f}, {time_emb.max():.3f}]")
    except Exception as e:
        print(f"Time embedding error: {e}")
        print("TODO: Implement time embedding")
    
    # TODO: Test forward pass
    try:
        batch_size = 4
        x = torch.randn(batch_size, 1, 32, 32)  # Grayscale 32x32 images
        t = torch.randint(0, 1000, (batch_size,))
        
        output = model(x, t)
        print(f"\nForward pass successful:")
        print(f"  Input shape: {x.shape}")
        print(f"  Timesteps shape: {t.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
    except Exception as e:
        print(f"\nForward pass error: {e}")
        print("TODO: Implement U-Net forward pass")
    
    # TODO: Analyze model
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Analysis:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"\nModel analysis error: {e}")


def exercise_3_ddpm_training():
    """
    Exercise 3: Implement DDPM training.
    
    Tasks:
    1. Complete DDPM loss computation
    2. Implement training loop
    3. Test on synthetic 2D data
    4. Monitor training progress
    """
    print("\nExercise 3: DDPM Training")
    print("=" * 50)
    
    # TODO: Create synthetic 2D dataset
    torch.manual_seed(42)
    n_samples = 1000
    
    # Create simple 2D mixture
    data1 = torch.randn(n_samples//2, 2) * 0.5 + torch.tensor([2., 2.])
    data2 = torch.randn(n_samples//2, 2) * 0.5 + torch.tensor([-2., -2.])
    synthetic_data = torch.cat([data1, data2], dim=0)
    
    # Add channel dimension for consistency with image data
    synthetic_data = synthetic_data.unsqueeze(1)  # Shape: (N, 1, 2)
    
    print(f"Created synthetic dataset: {synthetic_data.shape}")
    
    # TODO: Create DDPM components
    noise_scheduler = NoiseScheduler(num_timesteps=100)  # Fewer steps for 2D
    
    # Simple MLP instead of U-Net for 2D data
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=128, time_emb_dim=32):
            super().__init__()
            self.time_emb = nn.Sequential(
                nn.Linear(1, time_emb_dim),
                nn.ReLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
            
            self.net = nn.Sequential(
                nn.Linear(input_dim + time_emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
        
        def forward(self, x, t):
            # x shape: (batch, 1, 2) -> (batch, 2)
            x = x.squeeze(1)
            
            # Time embedding
            t_emb = self.time_emb(t.float().unsqueeze(-1))
            
            # Concatenate and process
            x_t = torch.cat([x, t_emb], dim=-1)
            out = self.net(x_t)
            
            return out.unsqueeze(1)  # Back to (batch, 1, 2)
    
    model = SimpleMLP()
    ddpm = DDPM(model, noise_scheduler)
    
    print("Created DDPM model")
    
    # TODO: Test loss computation
    try:
        test_batch = synthetic_data[:32]
        loss = ddpm.compute_loss(test_batch)
        print(f"Loss computation successful: {loss.item():.4f}")
    except Exception as e:
        print(f"Loss computation error: {e}")
        print("TODO: Implement DDPM loss")
    
    # TODO: Train DDPM
    try:
        dataset = TensorDataset(synthetic_data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        print("Starting DDPM training...")
        ddpm.train(dataloader, epochs=50, lr=0.001)
        
        print("Training completed")
        
    except Exception as e:
        print(f"Training error: {e}")
        print("TODO: Implement DDPM training")


def exercise_4_ddpm_sampling():
    """
    Exercise 4: Implement DDPM sampling.
    
    Tasks:
    1. Complete reverse diffusion sampling
    2. Implement DDIM sampling
    3. Compare sampling methods
    4. Visualize generation process
    """
    print("\nExercise 4: DDPM Sampling")
    print("=" * 50)
    
    # TODO: Use trained model from previous exercise
    # For demo, create a simple setup
    noise_scheduler = NoiseScheduler(num_timesteps=50)
    
    class DummyModel(nn.Module):
        def forward(self, x, t):
            # Dummy model that just returns small noise
            return 0.1 * torch.randn_like(x)
    
    model = DummyModel()
    ddpm = DDPM(model, noise_scheduler)
    
    print("Testing sampling methods:")
    
    # TODO: Test standard sampling
    try:
        samples = ddpm.p_sample(shape=(10, 1, 2))
        print(f"Standard sampling successful: {samples.shape}")
        print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    except Exception as e:
        print(f"Standard sampling error: {e}")
        print("TODO: Implement p_sample")
    
    # TODO: Test DDIM sampling
    try:
        sampler = DDPMSampler(ddpm)
        ddim_samples = sampler.ddim_sample(shape=(10, 1, 2), steps=10)
        print(f"DDIM sampling successful: {ddim_samples.shape}")
    except Exception as e:
        print(f"DDIM sampling error: {e}")
        print("TODO: Implement DDIM sampling")
    
    # TODO: Visualize sampling process
    print("\nTODO: Implement sampling visualization")


def exercise_5_advanced_features():
    """
    Exercise 5: Advanced DDPM features and analysis.
    
    Tasks:
    1. Implement conditional generation
    2. Study effect of different timestep samplings
    3. Analyze computational complexity
    4. Compare with other generative models
    """
    print("\nExercise 5: Advanced Features")
    print("=" * 50)
    
    print("Advanced DDPM Features:")
    print("1. Conditional Generation:")
    print("   - Class-conditional DDPM")
    print("   - Classifier guidance")
    print("   - Classifier-free guidance")
    
    print("\n2. Improved Sampling:")
    print("   - DDIM (faster deterministic)")
    print("   - DPM-Solver")
    print("   - Progressive distillation")
    
    print("\n3. Architecture Improvements:")
    print("   - Attention mechanisms")
    print("   - Better time embeddings")
    print("   - Skip connections")
    
    print("\n4. Training Improvements:")
    print("   - Importance sampling of timesteps")
    print("   - Loss weighting strategies")
    print("   - Variance reduction techniques")
    
    # TODO: Implement conditional generation
    class ConditionalUNet(nn.Module):
        def __init__(self, in_channels=1, num_classes=10):
            super().__init__()
            self.num_classes = num_classes
            # TODO: Add class embedding
            pass
        
        def forward(self, x, t, class_labels=None):
            # TODO: Incorporate class information
            pass
    
    print("\nTODO: Implement conditional generation")
    print("TODO: Implement advanced sampling methods")
    print("TODO: Implement training improvements")
    
    print("\nComputational Analysis:")
    print("- Training: O(T) forward passes per batch")
    print("- Sampling: O(T) denoising steps") 
    print("- Memory: Depends on model size, not T")
    print("- Parallelization: Batch generation possible")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("DDPM (Denoising Diffusion Probabilistic Models) Exercises")
    print("=" * 70)
    
    # Run exercises
    exercise_1_noise_scheduling()
    exercise_2_unet_architecture()
    exercise_3_ddpm_training()
    exercise_4_ddpm_sampling()
    exercise_5_advanced_features()
    
    print("\n" + "=" * 70)
    print("COMPLETION CHECKLIST:")
    print("✓ Implement noise scheduling (linear, cosine)")
    print("✓ Implement U-Net with time embedding")
    print("✓ Implement DDPM training loss")
    print("✓ Implement reverse diffusion sampling")
    print("✓ Implement DDIM fast sampling")
    print("✓ Analyze computational complexity")
    
    print("\nKey insights from DDPM:")
    print("- Diffusion models provide stable training")
    print("- Forward process is fixed, reverse is learned")
    print("- Quality-speed tradeoff in sampling")
    print("- Strong theoretical foundations")
    print("- Highly scalable to high-resolution images")
    print("- Connection to score-based models and SDEs") 