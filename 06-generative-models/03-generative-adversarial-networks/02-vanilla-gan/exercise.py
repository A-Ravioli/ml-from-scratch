"""
Vanilla GAN Implementation Exercise

Implement the original Generative Adversarial Network from the seminal paper
by Ian Goodfellow et al. (2014). Focus on understanding the minimax game theory,
training dynamics, and practical implementation challenges.

Key concepts:
- Minimax game between generator and discriminator
- Nash equilibrium and training stability
- Mode collapse and vanishing gradients
- Evaluation metrics for generative models

Author: ML-from-Scratch Course
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    """
    Generator network for Vanilla GAN.
    
    TODO: Implement a generator that maps noise to data distribution.
    
    Architecture guidelines:
    - Input: Random noise vector z ~ N(0, I)
    - Output: Generated samples in data space
    - Use fully connected layers with appropriate activations
    - Final layer should match data distribution (e.g., tanh for images)
    """
    
    def __init__(self, noise_dim: int = 100, hidden_dim: int = 256, 
                 output_dim: int = 784):
        """
        Initialize generator.
        
        Args:
            noise_dim: Dimension of input noise vector
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (e.g., 28*28 for MNIST)
        """
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        
        # TODO: Define generator architecture
        # Hint: Use nn.Sequential with Linear layers, BatchNorm, and activations
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


class Discriminator(nn.Module):
    """
    Discriminator network for Vanilla GAN.
    
    TODO: Implement a discriminator that distinguishes real from fake data.
    
    Architecture guidelines:
    - Input: Data samples (real or generated)
    - Output: Probability that input is real
    - Use fully connected layers with LeakyReLU activations
    - Final layer should output single probability
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256):
        """
        Initialize discriminator.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
        """
        super(Discriminator, self).__init__()
        
        # TODO: Define discriminator architecture
        # Hint: Use nn.Sequential with Linear layers and LeakyReLU
        self.model = None
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Forward pass through discriminator.
        
        Args:
            x: Input samples (batch_size, input_dim)
            
        Returns:
            Probabilities (batch_size, 1)
        """
        pass


class VanillaGAN:
    """
    Vanilla GAN implementation with training loop.
    
    TODO: Implement the complete GAN training procedure.
    """
    
    def __init__(self, generator: Generator, discriminator: Discriminator,
                 noise_dim: int = 100, device: str = 'cpu'):
        """
        Initialize Vanilla GAN.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
            noise_dim: Noise dimension
            device: Device for computation
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.noise_dim = noise_dim
        self.device = device
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'discriminator_real_acc': [],
            'discriminator_fake_acc': []
        }
    
    def train_step(self, real_data: torch.Tensor, 
                  g_optimizer: optim.Optimizer, d_optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        TODO: Implement single training step.
        
        Training procedure:
        1. Train discriminator on real data
        2. Train discriminator on fake data
        3. Train generator to fool discriminator
        
        Args:
            real_data: Batch of real data
            g_optimizer: Generator optimizer
            d_optimizer: Discriminator optimizer
            
        Returns:
            Dictionary with loss values and metrics
        """
        batch_size = real_data.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # TODO: Train Discriminator
        # Step 1: Train on real data
        # Step 2: Train on fake data
        # Hint: Zero gradients, compute loss, backward, step
        
        d_loss_real = None
        d_loss_fake = None
        d_loss = None
        
        # TODO: Train Generator
        # Generate fake data and train generator to fool discriminator
        # Hint: Use real_labels for generator loss (want discriminator to think fake is real)
        
        g_loss = None
        
        # TODO: Compute accuracies
        # Accuracy on real data: how often discriminator correctly identifies real as real
        # Accuracy on fake data: how often discriminator correctly identifies fake as fake
        
        d_real_acc = None
        d_fake_acc = None
        
        return {
            'g_loss': g_loss.item() if g_loss else 0.0,
            'd_loss': d_loss.item() if d_loss else 0.0,
            'd_real_acc': d_real_acc,
            'd_fake_acc': d_fake_acc
        }
    
    def train(self, dataloader: DataLoader, epochs: int = 100,
              g_lr: float = 0.0002, d_lr: float = 0.0002) -> None:
        """
        TODO: Implement full training loop.
        
        Args:
            dataloader: DataLoader for training data
            epochs: Number of training epochs
            g_lr: Generator learning rate
            d_lr: Discriminator learning rate
        """
        # TODO: Initialize optimizers
        # Hint: Use Adam optimizer with specified learning rates and beta1=0.5
        g_optimizer = None
        d_optimizer = None
        
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_d_real_acc = 0.0
            epoch_d_fake_acc = 0.0
            
            for batch_idx, (real_data, _) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                real_data = real_data.view(real_data.size(0), -1)  # Flatten
                
                # TODO: Perform training step
                metrics = self.train_step(real_data, g_optimizer, d_optimizer)
                
                # Accumulate metrics
                epoch_g_loss += metrics['g_loss']
                epoch_d_loss += metrics['d_loss']
                epoch_d_real_acc += metrics['d_real_acc']
                epoch_d_fake_acc += metrics['d_fake_acc']
            
            # Average metrics
            num_batches = len(dataloader)
            epoch_g_loss /= num_batches
            epoch_d_loss /= num_batches
            epoch_d_real_acc /= num_batches
            epoch_d_fake_acc /= num_batches
            
            # Store history
            self.history['generator_loss'].append(epoch_g_loss)
            self.history['discriminator_loss'].append(epoch_d_loss)
            self.history['discriminator_real_acc'].append(epoch_d_real_acc)
            self.history['discriminator_fake_acc'].append(epoch_d_fake_acc)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] "
                      f"G_Loss: {epoch_g_loss:.4f} "
                      f"D_Loss: {epoch_d_loss:.4f} "
                      f"D_Real_Acc: {epoch_d_real_acc:.4f} "
                      f"D_Fake_Acc: {epoch_d_fake_acc:.4f}")
    
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
        """TODO: Plot training metrics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Generator and Discriminator Loss
        axes[0, 0].plot(self.history['generator_loss'], label='Generator')
        axes[0, 0].plot(self.history['discriminator_loss'], label='Discriminator')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Discriminator Accuracies
        axes[0, 1].plot(self.history['discriminator_real_acc'], label='Real Data Acc')
        axes[0, 1].plot(self.history['discriminator_fake_acc'], label='Fake Data Acc')
        axes[0, 1].set_title('Discriminator Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss Ratio
        if len(self.history['generator_loss']) > 0 and len(self.history['discriminator_loss']) > 0:
            loss_ratio = np.array(self.history['generator_loss']) / (np.array(self.history['discriminator_loss']) + 1e-8)
            axes[1, 0].plot(loss_ratio)
            axes[1, 0].set_title('Generator/Discriminator Loss Ratio')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Ratio')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Balance indicator
        balance = np.array(self.history['discriminator_real_acc']) + np.array(self.history['discriminator_fake_acc'])
        axes[1, 1].plot(balance)
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Perfect Balance')
        axes[1, 1].set_title('Training Balance (Real + Fake Acc)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Sum of Accuracies')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def evaluate_gan_quality(generated_samples: torch.Tensor, real_samples: torch.Tensor) -> Dict[str, float]:
    """
    TODO: Implement basic GAN evaluation metrics.
    
    Args:
        generated_samples: Generated samples
        real_samples: Real samples
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    # TODO: Implement evaluation metrics
    # 1. Inception Score (simplified version)
    # 2. Mode coverage (how many modes are captured)
    # 3. Distribution distance (e.g., Wasserstein distance approximation)
    
    # Convert to numpy for easier computation
    gen_np = generated_samples.detach().cpu().numpy()
    real_np = real_samples.detach().cpu().numpy()
    
    # TODO: Sample diversity (standard deviation of generated samples)
    metrics['sample_diversity'] = np.std(gen_np)
    
    # TODO: Mean squared difference between real and generated distributions
    # Compute histograms and compare
    
    return metrics


def visualize_generated_samples(generator: Generator, noise_dim: int, 
                               device: str, num_samples: int = 64,
                               image_shape: Tuple[int, int] = (28, 28)) -> None:
    """
    TODO: Visualize generated samples in a grid.
    
    Args:
        generator: Trained generator
        noise_dim: Noise dimension
        device: Device
        num_samples: Number of samples to generate
        image_shape: Shape to reshape samples (for images)
    """
    generator.eval()
    with torch.no_grad():
        # TODO: Generate samples
        noise = torch.randn(num_samples, noise_dim, device=device)
        generated = generator(noise)
        generated = generated.cpu()
        
        # TODO: Reshape and visualize
        # Hint: Use matplotlib subplots to create a grid
        grid_size = int(np.sqrt(num_samples))
        
        pass


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_implement_networks():
    """
    Exercise 1: Implement Generator and Discriminator networks.
    
    Tasks:
    1. Complete the Generator class with appropriate architecture
    2. Complete the Discriminator class with appropriate architecture
    3. Test forward passes with dummy data
    4. Verify output shapes and ranges
    """
    print("Exercise 1: Implementing Generator and Discriminator")
    print("=" * 50)
    
    # TODO: Create generator and discriminator
    generator = Generator(noise_dim=100, hidden_dim=256, output_dim=784)
    discriminator = Discriminator(input_dim=784, hidden_dim=256)
    
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
    
    print("\nTesting Discriminator:")
    try:
        disc_output = discriminator(fake_data)
        print(f"Discriminator output shape: {disc_output.shape}")
        print(f"Discriminator output range: [{disc_output.min():.3f}, {disc_output.max():.3f}]")
    except Exception as e:
        print(f"Discriminator error: {e}")


def exercise_2_training_loop():
    """
    Exercise 2: Implement and test the training loop.
    
    Tasks:
    1. Complete the train_step method
    2. Complete the full training loop
    3. Test on synthetic 2D data
    4. Monitor training stability
    """
    print("\nExercise 2: GAN Training Loop")
    print("=" * 50)
    
    # Create synthetic 2D data (mixture of Gaussians)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # TODO: Generate synthetic 2D dataset
    # Hint: Create mixture of 2-3 Gaussian distributions
    n_samples = 1000
    data = []
    
    # Component 1
    data.append(np.random.normal([2, 2], 0.5, (n_samples//3, 2)))
    # Component 2  
    data.append(np.random.normal([-2, -2], 0.5, (n_samples//3, 2)))
    # Component 3
    data.append(np.random.normal([2, -2], 0.5, (n_samples//3, 2)))
    
    synthetic_data = np.vstack(data)
    
    # Convert to PyTorch
    dataset = TensorDataset(torch.FloatTensor(synthetic_data), torch.zeros(len(synthetic_data)))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # TODO: Create and train GAN
    generator = Generator(noise_dim=10, hidden_dim=32, output_dim=2)
    discriminator = Discriminator(input_dim=2, hidden_dim=32)
    
    gan = VanillaGAN(generator, discriminator, noise_dim=10)
    
    print("Starting training...")
    gan.train(dataloader, epochs=50, g_lr=0.001, d_lr=0.001)
    
    # TODO: Plot results
    gan.plot_training_history()
    
    # TODO: Generate and visualize samples
    generated_samples = gan.generate_samples(500)
    
    if generated_samples is not None:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], alpha=0.6, label='Real Data')
        plt.title('Real Data Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        gen_np = generated_samples.detach().cpu().numpy()
        plt.scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.6, label='Generated Data', color='red')
        plt.title('Generated Data Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


def exercise_3_mnist_gan():
    """
    Exercise 3: Train GAN on MNIST dataset.
    
    Tasks:
    1. Load and preprocess MNIST data
    2. Train GAN on MNIST
    3. Generate and visualize MNIST-like digits
    4. Evaluate generation quality
    """
    print("\nExercise 3: MNIST GAN")
    print("=" * 50)
    
    try:
        from torchvision import datasets, transforms
        
        # TODO: Load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
        
        # TODO: Create GAN for MNIST
        generator = Generator(noise_dim=100, hidden_dim=256, output_dim=784)
        discriminator = Discriminator(input_dim=784, hidden_dim=256)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gan = VanillaGAN(generator, discriminator, noise_dim=100, device=device)
        
        print(f"Training on device: {device}")
        print("Starting MNIST GAN training...")
        
        # Train for fewer epochs for demo
        gan.train(dataloader, epochs=20, g_lr=0.0002, d_lr=0.0002)
        
        # TODO: Generate and visualize results
        gan.plot_training_history()
        visualize_generated_samples(gan.generator, 100, device, num_samples=64)
        
    except ImportError:
        print("Torchvision not available. Skipping MNIST example.")
        print("Install with: pip install torchvision")


def exercise_4_analyze_training_dynamics():
    """
    Exercise 4: Analyze GAN training dynamics and stability.
    
    Tasks:
    1. Experiment with different learning rates
    2. Analyze mode collapse scenarios
    3. Study discriminator-generator balance
    4. Implement techniques to improve stability
    """
    print("\nExercise 4: Training Dynamics Analysis")
    print("=" * 50)
    
    # TODO: Create experiments with different hyperparameters
    experiments = [
        {'g_lr': 0.0002, 'd_lr': 0.0002, 'name': 'Balanced'},
        {'g_lr': 0.001, 'd_lr': 0.0002, 'name': 'Strong Generator'},
        {'g_lr': 0.0002, 'd_lr': 0.001, 'name': 'Strong Discriminator'},
    ]
    
    # Simple 1D dataset for quick experiments
    np.random.seed(42)
    real_data = np.random.normal(2, 1, (1000, 1))
    dataset = TensorDataset(torch.FloatTensor(real_data), torch.zeros(len(real_data)))
    
    results = {}
    
    for exp in experiments:
        print(f"\nRunning experiment: {exp['name']}")
        
        # Create fresh networks
        generator = Generator(noise_dim=10, hidden_dim=32, output_dim=1)
        discriminator = Discriminator(input_dim=1, hidden_dim=32)
        
        gan = VanillaGAN(generator, discriminator, noise_dim=10)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train with specific hyperparameters
        gan.train(dataloader, epochs=100, g_lr=exp['g_lr'], d_lr=exp['d_lr'])
        
        results[exp['name']] = gan.history
    
    # TODO: Compare training curves
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(['generator_loss', 'discriminator_loss', 'discriminator_real_acc']):
        plt.subplot(1, 3, i+1)
        for name, history in results.items():
            plt.plot(history[metric], label=name)
        plt.title(metric.replace('_', ' ').title())
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis complete!")
    print("Key observations:")
    print("- Strong generator: May lead to discriminator collapse")
    print("- Strong discriminator: May prevent generator learning")
    print("- Balanced learning rates often work best")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Vanilla GAN Implementation Exercises")
    print("=" * 60)
    
    # Run exercises
    exercise_1_implement_networks()
    exercise_2_training_loop()
    exercise_3_mnist_gan()
    exercise_4_analyze_training_dynamics()
    
    print("\n" + "=" * 60)
    print("COMPLETION CHECKLIST:")
    print("✓ Implement Generator and Discriminator architectures")
    print("✓ Implement GAN training loop with proper loss functions")
    print("✓ Test on synthetic 2D data")
    print("✓ Train on MNIST dataset")
    print("✓ Implement visualization and evaluation tools")
    print("✓ Analyze training dynamics and stability")
    
    print("\nKey insights from Vanilla GAN:")
    print("- Training GANs requires careful balance between G and D")
    print("- Mode collapse is a common issue")
    print("- Evaluation of generative models is challenging")
    print("- Hyperparameter tuning is crucial for stability") 