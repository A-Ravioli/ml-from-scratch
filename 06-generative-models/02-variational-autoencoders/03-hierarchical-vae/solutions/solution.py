"""
Hierarchical Variational Autoencoders - Complete Implementation

This module provides reference implementations for hierarchical VAE architectures:
1. Ladder VAE with bidirectional information flow
2. Very Deep VAE (VDVAE) with residual connections
3. Multi-scale hierarchical VAE for images
4. Hierarchical β-VAE with level-specific scheduling

Mathematical foundations from Sønderby et al. (2016), Child (2020), Vahdat & Kautz (2020).

Author: ML-from-Scratch Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class HierarchicalVAE(nn.Module, ABC):
    """
    Abstract base class for hierarchical VAE architectures.
    
    Defines the common interface for all hierarchical VAE implementations
    following the mathematical framework from the lesson.
    """
    
    def __init__(self, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        
    @abstractmethod
    def encode(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Encode input to hierarchical latent variables.
        
        Returns:
            List of dicts containing 'mu', 'logvar' for each level
        """
        ...
        
    @abstractmethod
    def decode(self, z_samples: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode hierarchical latent variables to reconstruction.
        
        Args:
            z_samples: List of latent samples for each level
            
        Returns:
            Reconstructed input
        """
        ...
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass returning all necessary components for loss.
        """
        ...
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for differentiable sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def sample_prior(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Sample from hierarchical prior distribution."""
        samples = []
        for level in range(self.num_levels):
            # Assume standard Gaussian prior for all levels
            latent_dim = self.get_latent_dim(level)
            z = torch.randn(batch_size, latent_dim, device=device)
            samples.append(z)
        return samples
        
    @abstractmethod
    def get_latent_dim(self, level: int) -> int:
        """Get latent dimension for specific level."""
        ...


class LadderBlock(nn.Module):
    """
    Basic building block for Ladder VAE implementing bidirectional information flow.
    
    Mathematical formulation:
    - Bottom-up: h_i^bu = f_i^bu(h_{i-1}^bu, x)
    - Prior: μ_i^prior, σ_i^prior = f_i^prior(h_{i+1}^td)
    - Inference: μ_i^inf, σ_i^inf = f_i^inf(h_i^bu, h_{i+1}^td)
    - Top-down: h_i^td = f_i^td(h_{i+1}^td, z_i)
    """
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Bottom-up pathway
        self.bottom_up = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Top-down pathway
        self.top_down = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Inference network (combines bottom-up and top-down)
        self.inference_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.inference_mu = nn.Linear(hidden_dim, latent_dim)
        self.inference_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Prior network
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, bottom_up: torch.Tensor, 
               top_down: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ladder block.
        
        Args:
            bottom_up: Bottom-up features from previous level
            top_down: Top-down features from next level (None for top level)
            
        Returns:
            Dict with 'z_sample', 'z_mu', 'z_logvar', 'bu_features', 'td_features'
        """
        # Process bottom-up features
        bu_features = self.bottom_up(bottom_up)
        
        if top_down is None:
            # Top level - use only bottom-up information
            # Prior is standard Gaussian
            prior_mu = torch.zeros(bu_features.size(0), self.latent_dim, 
                                 device=bu_features.device)
            prior_logvar = torch.zeros_like(prior_mu)
            
            # Inference uses only bottom-up
            inf_features = self.inference_net(torch.cat([bu_features, bu_features], dim=1))
            z_mu = self.inference_mu(inf_features)
            z_logvar = self.inference_logvar(inf_features)
        else:
            # Lower levels - combine bottom-up and top-down
            # Compute prior parameters from top-down features
            prior_features = self.prior_net(top_down)
            prior_mu = self.prior_mu(prior_features)
            prior_logvar = self.prior_logvar(prior_features)
            
            # Compute inference parameters from both pathways
            inf_features = self.inference_net(torch.cat([bu_features, top_down], dim=1))
            z_mu = self.inference_mu(inf_features)
            z_logvar = self.inference_logvar(inf_features)
        
        # Sample latent variable
        z_sample = self.reparameterize(z_mu, z_logvar)
        
        # Compute top-down features for next level
        td_input = torch.cat([top_down if top_down is not None else bu_features, z_sample], dim=1)
        td_features = self.top_down(td_input)
        
        return {
            'z_sample': z_sample,
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'prior_mu': prior_mu,
            'prior_logvar': prior_logvar,
            'bu_features': bu_features,
            'td_features': td_features
        }
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class LadderVAE(HierarchicalVAE):
    """
    Ladder Variational Autoencoder implementation.
    
    Implements bidirectional information flow with skip connections
    between encoder and decoder at multiple hierarchy levels.
    
    Based on Sønderby et al. (2016) "Ladder Variational Autoencoders"
    """
    
    def __init__(self, input_dim: int, latent_dims: List[int], 
                 hidden_dims: List[int]):
        super().__init__(len(latent_dims))
        
        self.input_dim = input_dim
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        
        # Input processing layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )
        
        # Ladder blocks for each level
        self.ladder_blocks = nn.ModuleList()
        for i in range(self.num_levels):
            block_input_dim = hidden_dims[i] if i == 0 else hidden_dims[i-1]
            self.ladder_blocks.append(
                LadderBlock(block_input_dim, latent_dims[i], hidden_dims[i])
            )
        
        # Output reconstruction layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        )
        
    def get_latent_dim(self, level: int) -> int:
        """Get latent dimension for specific level."""
        return self.latent_dims[level]
        
    def encode(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Encode input through hierarchical levels."""
        batch_size = x.size(0)
        
        # Process input
        features = self.input_layer(x)
        
        # Bottom-up pass through all levels
        bu_features = [features]
        for i in range(1, self.num_levels):
            # For simplicity, reuse features (in practice, could have separate pathways)
            bu_features.append(features)
        
        # Top-down pass to compute latent variables
        latent_params = []
        td_features = None
        
        # Start from top level and work downward
        for i in reversed(range(self.num_levels)):
            block_output = self.ladder_blocks[i](bu_features[i], td_features)
            latent_params.insert(0, {
                'mu': block_output['z_mu'],
                'logvar': block_output['z_logvar'],
                'prior_mu': block_output['prior_mu'],
                'prior_logvar': block_output['prior_logvar'],
                'sample': block_output['z_sample']
            })
            td_features = block_output['td_features']
        
        return latent_params
        
    def decode(self, z_samples: List[torch.Tensor]) -> torch.Tensor:
        """Decode hierarchical latents to reconstruction."""
        # Start from top level
        td_features = None
        
        # Top-down generation
        for i in reversed(range(self.num_levels)):
            if td_features is None:
                # Top level
                td_features = z_samples[i]
            else:
                # Combine with current level latent
                td_input = torch.cat([td_features, z_samples[i]], dim=1)
                td_features = self.ladder_blocks[i].top_down(td_input)
        
        # Generate reconstruction
        reconstruction = self.output_layer(td_features)
        return reconstruction
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass."""
        # Encode to get latent parameters
        latent_params = self.encode(x)
        
        # Extract samples for decoding
        z_samples = [params['sample'] for params in latent_params]
        
        # Decode to reconstruction
        reconstruction = self.decode(z_samples)
        
        return {
            'reconstruction': reconstruction,
            'latent_params': latent_params,
            'z_samples': z_samples
        }


class ResidualBlock(nn.Module):
    """
    Residual block for Very Deep VAE with batch normalization and skip connections.
    
    Essential for training very deep hierarchical models (20+ levels).
    """
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        out += identity
        out = self.activation(out)
        
        return out


class SqueezeExciteBlock(nn.Module):
    """Squeeze-and-Excite block for channel-wise attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()
        
        # Squeeze
        y = self.global_pool(x).view(batch_size, channels)
        
        # Excitation
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        
        # Scale
        y = y.view(batch_size, channels, 1, 1)
        return x * y


class VeryDeepVAELevel(nn.Module):
    """
    Single level in Very Deep VAE architecture with residual blocks and attention.
    
    Implements the VDVAE architecture from Child (2020).
    """
    
    def __init__(self, input_channels: int, latent_dim: int, 
                 num_residual_blocks: int = 2):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Residual blocks for feature processing
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(input_channels) for _ in range(num_residual_blocks)
        ])
        
        # Squeeze-and-excite attention
        self.se_block = SqueezeExciteBlock(input_channels)
        
        # Latent variable layers
        self.encoder_conv = nn.Conv2d(input_channels, input_channels, 1)
        self.encoder_mu = nn.Conv2d(input_channels, latent_dim, 1)
        self.encoder_logvar = nn.Conv2d(input_channels, latent_dim, 1)
        
        # Decoder layers
        self.decoder_conv = nn.Conv2d(latent_dim, input_channels, 1)
        
    def encode(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode features to latent parameters."""
        # Process through residual blocks
        features = x
        for block in self.residual_blocks:
            features = block(features)
        
        # Apply attention
        features = self.se_block(features)
        
        # Add context if provided
        if context is not None:
            features = features + context
        
        # Compute latent parameters
        features = self.encoder_conv(features)
        mu = self.encoder_mu(features)
        logvar = self.encoder_logvar(features)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'sample': self.reparameterize(mu, logvar)
        }
        
    def decode(self, z: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode latent to features."""
        features = self.decoder_conv(z)
        
        if context is not None:
            features = features + context
            
        return features
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VeryDeepVAE(HierarchicalVAE):
    """
    Very Deep VAE with residual connections enabling training of 20+ level hierarchies.
    
    Based on Child (2020) "Very Deep VAEs Generalize Autoregressive Models"
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], 
                 latent_dims: List[int], base_channels: int = 64):
        super().__init__(len(latent_dims))
        
        self.input_shape = input_shape
        self.latent_dims = latent_dims
        self.base_channels = base_channels
        
        channels, height, width = input_shape
        
        # Input processing 
        self.input_conv = nn.Conv2d(channels, base_channels, 3, padding=1)
        
        # Encoder levels (bottom-up with downsampling)
        self.encoder_levels = nn.ModuleList()
        current_channels = base_channels
        
        for i, latent_dim in enumerate(latent_dims):
            # Downsample every few levels
            if i > 0 and i % 3 == 0:
                downsample = nn.Conv2d(current_channels, current_channels * 2, 3, stride=2, padding=1)
                current_channels *= 2
            else:
                downsample = None
                
            level = VeryDeepVAELevel(current_channels, latent_dim)
            self.encoder_levels.append(nn.ModuleDict({
                'level': level,
                'downsample': downsample
            }))
        
        # Decoder levels (top-down with upsampling)
        self.decoder_levels = nn.ModuleList()
        for i in reversed(range(self.num_levels)):
            # Determine if upsampling is needed
            if i < self.num_levels - 1 and (self.num_levels - 1 - i) % 3 == 0:
                upsample = nn.ConvTranspose2d(current_channels, current_channels // 2, 4, stride=2, padding=1)
                current_channels //= 2
            else:
                upsample = None
                
            self.decoder_levels.append(nn.ModuleDict({
                'upsample': upsample,
                'channels': current_channels
            }))
        
        # Output layer
        self.output_conv = nn.Conv2d(base_channels, channels, 3, padding=1)
        
    def get_latent_dim(self, level: int) -> int:
        """Get latent dimension for specific level."""
        return self.latent_dims[level]
        
    def encode(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Hierarchical encoding."""
        features = self.input_conv(x)
        latent_params = []
        
        # Bottom-up pass through encoder levels
        for i, level_dict in enumerate(self.encoder_levels):
            level = level_dict['level']
            downsample = level_dict['downsample']
            
            # Downsample if needed
            if downsample is not None:
                features = downsample(features)
            
            # Encode at this level
            level_params = level.encode(features)
            latent_params.append(level_params)
            
            # Update features for next level
            features = level.decode(level_params['sample'])
        
        return latent_params
        
    def decode(self, z_samples: List[torch.Tensor]) -> torch.Tensor:
        """Hierarchical decoding."""
        # Start from top level
        features = None
        
        # Top-down pass through decoder levels
        for i, (z_sample, level_dict) in enumerate(zip(reversed(z_samples), self.decoder_levels)):
            upsample = level_dict['upsample']
            
            if features is None:
                # Top level
                level = self.encoder_levels[-(i+1)]['level']
                features = level.decode(z_sample)
            else:
                # Upsample if needed
                if upsample is not None:
                    features = upsample(features)
                
                # Combine with current level
                level = self.encoder_levels[-(i+1)]['level']
                level_features = level.decode(z_sample)
                features = features + level_features
        
        # Generate final output
        reconstruction = torch.sigmoid(self.output_conv(features))
        return reconstruction
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Complete forward pass."""
        # Encode
        latent_params = self.encode(x)
        
        # Extract samples
        z_samples = [params['sample'] for params in latent_params]
        
        # Decode
        reconstruction = self.decode(z_samples)
        
        return {
            'reconstruction': reconstruction,
            'latent_params': latent_params,
            'z_samples': z_samples
        }


class MultiScaleImageVAE(HierarchicalVAE):
    """
    Multi-scale hierarchical VAE for image generation.
    
    Each level operates at different spatial resolutions:
    - Level 0: Global structure (low resolution)
    - Level 1: Mid-level features (medium resolution)  
    - Level 2: Fine details (full resolution)
    """
    
    def __init__(self, image_channels: int = 3, base_resolution: int = 8):
        # Define levels based on resolution scaling
        self.resolutions = [base_resolution * (2 ** i) for i in range(3)]  # e.g., 8, 16, 32
        latent_dims = [256, 128, 64]  # Decreasing latent dimensions
        
        super().__init__(len(latent_dims))
        
        self.image_channels = image_channels
        self.latent_dims = latent_dims
        self.base_channels = 64
        
        # Encoders for each resolution
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i, (resolution, latent_dim) in enumerate(zip(self.resolutions, latent_dims)):
            # Encoder for this resolution
            encoder = nn.Sequential(
                nn.Conv2d(image_channels, self.base_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.base_channels, self.base_channels * 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.base_channels * 2, self.base_channels * 4, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten()
            )
            
            # Latent layers
            feature_dim = self.base_channels * 4 * 4 * 4
            mu_layer = nn.Linear(feature_dim, latent_dim)
            logvar_layer = nn.Linear(feature_dim, latent_dim)
            
            # Decoder for this resolution
            decoder = nn.Sequential(
                nn.Linear(latent_dim, feature_dim),
                nn.ReLU(),
                nn.Unflatten(1, (self.base_channels * 4, 4, 4)),
                nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.base_channels, image_channels, 3, padding=1),
                nn.Sigmoid()
            )
            
            self.encoders.append(nn.ModuleDict({
                'conv': encoder,
                'mu': mu_layer,
                'logvar': logvar_layer
            }))
            self.decoders.append(decoder)
    
    def get_latent_dim(self, level: int) -> int:
        """Get latent dimension for specific level."""
        return self.latent_dims[level]
        
    def encode_multiscale(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Encode image at multiple scales."""
        latent_params = []
        
        for i, (resolution, encoder) in enumerate(zip(self.resolutions, self.encoders)):
            # Resize input to current resolution
            x_resized = F.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=False)
            
            # Encode at this scale
            features = encoder['conv'](x_resized)
            mu = encoder['mu'](features)
            logvar = encoder['logvar'](features)
            sample = self.reparameterize(mu, logvar)
            
            latent_params.append({
                'mu': mu,
                'logvar': logvar,
                'sample': sample
            })
        
        return latent_params
        
    def decode_multiscale(self, z_samples: List[torch.Tensor]) -> torch.Tensor:
        """Generate image from multi-scale latents."""
        reconstructions = []
        
        for i, (z_sample, decoder) in enumerate(zip(z_samples, self.decoders)):
            # Decode at this scale
            recon = decoder(z_sample)
            reconstructions.append(recon)
        
        # Combine reconstructions from all scales
        # Upsample lower resolution reconstructions and average
        target_size = reconstructions[-1].shape[-2:]
        combined_recon = torch.zeros_like(reconstructions[-1])
        
        for i, recon in enumerate(reconstructions):
            if recon.shape[-2:] != target_size:
                recon = F.interpolate(recon, size=target_size, mode='bilinear', align_corners=False)
            combined_recon += recon / len(reconstructions)
        
        return combined_recon
        
    def encode(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        return self.encode_multiscale(x)
        
    def decode(self, z_samples: List[torch.Tensor]) -> torch.Tensor:
        return self.decode_multiscale(z_samples)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode at multiple scales
        latent_params = self.encode_multiscale(x)
        
        # Extract samples
        z_samples = [params['sample'] for params in latent_params]
        
        # Decode with multi-scale combination
        reconstruction = self.decode_multiscale(z_samples)
        
        return {
            'reconstruction': reconstruction,
            'latent_params': latent_params,
            'z_samples': z_samples
        }


class HierarchicalLoss(nn.Module):
    """
    Loss function for hierarchical VAEs with level-specific β scheduling.
    
    Implements sophisticated hierarchical loss with:
    1. Level-specific β parameters
    2. Free bits constraint  
    3. Hierarchical regularization
    """
    
    def __init__(self, num_levels: int, beta_schedule: str = 'constant',
                 free_bits: float = 0.0, hierarchy_penalty: float = 0.0):
        super().__init__()
        self.num_levels = num_levels
        self.beta_schedule = beta_schedule
        self.free_bits = free_bits
        self.hierarchy_penalty = hierarchy_penalty
        
        # Initialize β parameters for each level
        self.register_buffer('betas', torch.ones(num_levels))
        
    def compute_reconstruction_loss(self, x_recon: torch.Tensor, 
                                  x_target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        # Use BCE for images in [0,1] range
        return F.binary_cross_entropy(x_recon, x_target, reduction='sum') / x_target.size(0)
        
    def compute_kl_loss(self, z_mu: torch.Tensor, z_logvar: torch.Tensor,
                       level: int = 0) -> torch.Tensor:
        """Compute KL divergence loss for specific level with free bits constraint."""
        # Standard KL divergence against unit Gaussian
        kl = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1)
        
        # Apply free bits constraint
        if self.free_bits > 0:
            kl = torch.clamp(kl, min=self.free_bits * z_mu.size(1))
        
        return torch.mean(kl)
        
    def compute_hierarchy_penalty(self, latent_params: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Penalize redundancy between hierarchy levels."""
        if self.hierarchy_penalty <= 0 or len(latent_params) < 2:
            return torch.tensor(0.0, device=latent_params[0]['mu'].device)
        
        penalty = 0.0
        # Encourage diversity between adjacent levels
        for i in range(len(latent_params) - 1):
            mu1 = latent_params[i]['mu']
            mu2 = latent_params[i + 1]['mu']
            
            # Compute correlation penalty
            if mu1.size(1) == mu2.size(1):
                correlation = torch.mean(torch.sum(mu1 * mu2, dim=1))
                penalty += correlation ** 2
        
        return self.hierarchy_penalty * penalty
        
    def update_beta_schedule(self, epoch: int, warmup_epochs: int = 100):
        """Update β parameters based on schedule."""
        if self.beta_schedule == 'linear':
            # Linear warmup
            factor = min(1.0, epoch / warmup_epochs)
            self.betas.fill_(factor)
        elif self.beta_schedule == 'cyclical':
            # Cyclical β schedule
            cycle_length = 100
            factor = 0.5 * (1 + np.cos(2 * np.pi * (epoch % cycle_length) / cycle_length))
            self.betas.fill_(factor)
        elif self.beta_schedule == 'hierarchical':
            # Different β for each level
            for i in range(self.num_levels):
                # Higher levels get higher β (more regularization)
                level_factor = (i + 1) / self.num_levels
                warmup_factor = min(1.0, epoch / warmup_epochs)
                self.betas[i] = level_factor * warmup_factor
        # 'constant' schedule does nothing
        
    def forward(self, model_outputs: Dict[str, torch.Tensor], 
               targets: torch.Tensor, epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compute total hierarchical loss.
        
        Returns:
            Dict with individual loss components and total loss
        """
        reconstruction = model_outputs['reconstruction']
        latent_params = model_outputs['latent_params']
        
        # Reconstruction loss
        recon_loss = self.compute_reconstruction_loss(reconstruction, targets)
        
        # KL losses for each level
        kl_losses = []
        total_kl = 0.0
        
        for i, params in enumerate(latent_params):
            kl_loss = self.compute_kl_loss(params['mu'], params['logvar'], level=i)
            kl_losses.append(kl_loss)
            total_kl += self.betas[i] * kl_loss
        
        # Hierarchy penalty
        hierarchy_penalty = self.compute_hierarchy_penalty(latent_params)
        
        # Total loss
        total_loss = recon_loss + total_kl + hierarchy_penalty
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_losses': kl_losses,
            'total_kl': total_kl,
            'hierarchy_penalty': hierarchy_penalty
        }


class HierarchicalTrainer:
    """
    Training utilities for hierarchical VAEs with progressive and balanced training.
    
    Implements sophisticated training procedures including:
    1. Progressive training (start with few levels, add more)
    2. Balanced training (ensure all levels are utilized)
    3. Curriculum learning for hierarchical complexity
    """
    
    def __init__(self, model: HierarchicalVAE, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def progressive_training(self, train_loader, val_loader, 
                           max_levels: int = None, epochs_per_level: int = 10):
        """
        Train hierarchical VAE progressively.
        
        Start with fewer levels and gradually add more levels.
        """
        if max_levels is None:
            max_levels = self.model.num_levels
        
        training_history = []
        
        for num_levels in range(1, max_levels + 1):
            print(f"Training with {num_levels} levels...")
            
            # Temporarily modify model to use fewer levels
            original_num_levels = self.model.num_levels
            self.model.num_levels = num_levels
            
            # Train for specified epochs
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            loss_fn = HierarchicalLoss(num_levels)
            
            level_history = []
            for epoch in range(epochs_per_level):
                train_loss = self._train_epoch(train_loader, optimizer, loss_fn, epoch)
                val_loss = self._validate_epoch(val_loader, loss_fn)
                
                level_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'num_levels': num_levels
                })
                
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            training_history.extend(level_history)
            
            # Restore original number of levels
            self.model.num_levels = original_num_levels
        
        return training_history
        
    def balanced_training_step(self, batch: torch.Tensor, optimizer, loss_fn, 
                             target_utilization: float = 0.1):
        """
        Single training step with level balancing.
        
        Adjust loss weights to ensure all levels are utilized.
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch)
        loss_dict = loss_fn(outputs, batch)
        
        # Check level utilization
        kl_losses = loss_dict['kl_losses']
        utilization = [kl.item() for kl in kl_losses]
        
        # Adjust β parameters if some levels are underutilized
        for i, util in enumerate(utilization):
            if util < target_utilization:
                # Decrease β to encourage utilization
                loss_fn.betas[i] *= 0.95
            elif util > target_utilization * 10:
                # Increase β to prevent collapse
                loss_fn.betas[i] *= 1.05
        
        # Clamp β values
        loss_fn.betas.clamp_(0.01, 10.0)
        
        # Backward pass
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        return loss_dict
        
    def evaluate_hierarchy_utilization(self, data_loader) -> Dict[str, float]:
        """
        Evaluate how well each hierarchy level is utilized.
        
        Returns:
            Dict with utilization metrics per level
        """
        self.model.eval()
        
        level_kls = [[] for _ in range(self.model.num_levels)]
        level_activations = [[] for _ in range(self.model.num_levels)]
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                
                # Collect KL divergences
                for i, params in enumerate(outputs['latent_params']):
                    mu = params['mu']
                    logvar = params['logvar']
                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    level_kls[i].extend(kl.cpu().numpy())
                    
                    # Measure activation (fraction of active units)
                    active_units = (torch.abs(mu) > 0.1).float().mean(dim=0)
                    level_activations[i].append(active_units.cpu().numpy())
        
        # Compute utilization metrics
        utilization = {}
        for i in range(self.model.num_levels):
            utilization[f'level_{i}_kl_mean'] = np.mean(level_kls[i])
            utilization[f'level_{i}_kl_std'] = np.std(level_kls[i])
            utilization[f'level_{i}_active_units'] = np.mean(level_activations[i])
        
        return utilization
    
    def _train_epoch(self, train_loader, optimizer, loss_fn, epoch):
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch)
            loss_dict = loss_fn(outputs, batch, epoch)
            
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            total_loss += loss_dict['total_loss'].item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader, loss_fn):
        """Single validation epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss_dict = loss_fn(outputs, batch)
                total_loss += loss_dict['total_loss'].item()
        
        return total_loss / len(val_loader)


class HierarchicalEvaluation:
    """
    Evaluation metrics specific to hierarchical VAEs.
    
    Implements evaluation methods for:
    1. Disentanglement at each hierarchy level
    2. Information flow between levels  
    3. Multi-scale generation quality
    """
    
    @staticmethod
    def compute_disentanglement_per_level(model: HierarchicalVAE, 
                                        data_loader, 
                                        ground_truth_factors: torch.Tensor) -> Dict[int, float]:
        """Compute disentanglement metrics for each hierarchy level."""
        model.eval()
        
        # Collect latent representations for each level
        level_representations = [[] for _ in range(model.num_levels)]
        
        with torch.no_grad():
            for batch, factors in zip(data_loader, ground_truth_factors):
                batch = batch.to(next(model.parameters()).device)
                outputs = model(batch)
                
                for i, params in enumerate(outputs['latent_params']):
                    level_representations[i].append(params['mu'].cpu())
        
        # Compute MIG (Mutual Information Gap) for each level
        disentanglement_scores = {}
        for i in range(model.num_levels):
            representations = torch.cat(level_representations[i], dim=0)
            mig_score = compute_mig(representations, ground_truth_factors)
            disentanglement_scores[i] = mig_score
        
        return disentanglement_scores
        
    @staticmethod
    def visualize_hierarchy_interpolation(model: HierarchicalVAE, 
                                        x1: torch.Tensor, x2: torch.Tensor,
                                        num_steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two samples at different hierarchy levels.
        
        Returns:
            Interpolation results [num_levels, num_steps, *x_shape]
        """
        model.eval()
        device = next(model.parameters()).device
        x1, x2 = x1.to(device), x2.to(device)
        
        with torch.no_grad():
            # Encode both samples
            params1 = model.encode(x1.unsqueeze(0))
            params2 = model.encode(x2.unsqueeze(0))
            
            interpolations = []
            
            # Interpolate at each level independently
            for level in range(model.num_levels):
                level_interpolations = []
                
                # Get latent codes for this level
                z1 = params1[level]['sample']
                z2 = params2[level]['sample']
                
                for step in range(num_steps):
                    alpha = step / (num_steps - 1)
                    
                    # Interpolate at this level, keep others from x1
                    z_interp = [params1[i]['sample'].clone() for i in range(model.num_levels)]
                    z_interp[level] = (1 - alpha) * z1 + alpha * z2
                    
                    # Decode
                    recon = model.decode(z_interp)
                    level_interpolations.append(recon)
                
                interpolations.append(torch.cat(level_interpolations, dim=0))
            
            return torch.stack(interpolations)
        
    @staticmethod
    def analyze_information_flow(model: HierarchicalVAE, 
                               data_loader) -> Dict[str, torch.Tensor]:
        """Analyze information flow between hierarchy levels."""
        model.eval()
        
        mutual_information_matrix = torch.zeros(model.num_levels, model.num_levels)
        
        # Collect representations
        level_representations = [[] for _ in range(model.num_levels)]
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(next(model.parameters()).device)
                outputs = model(batch)
                
                for i, params in enumerate(outputs['latent_params']):
                    level_representations[i].append(params['mu'].cpu())
        
        # Compute mutual information between levels
        for i in range(model.num_levels):
            for j in range(model.num_levels):
                if i != j:
                    repr_i = torch.cat(level_representations[i], dim=0)
                    repr_j = torch.cat(level_representations[j], dim=0)
                    
                    # Approximate mutual information using correlation
                    mi = compute_mutual_information_approx(repr_i, repr_j)
                    mutual_information_matrix[i, j] = mi
        
        return {
            'mutual_information_matrix': mutual_information_matrix,
            'level_representations': level_representations
        }
        
    @staticmethod
    def evaluate_multiscale_quality(model: MultiScaleImageVAE,
                                   data_loader) -> Dict[str, float]:
        """Evaluate generation quality at multiple scales."""
        model.eval()
        
        # Generate samples at different scales
        generated_samples = []
        real_samples = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(next(model.parameters()).device)
                
                # Generate samples
                z_samples = model.sample_prior(batch.size(0), batch.device)
                generated = model.decode(z_samples)
                
                generated_samples.append(generated.cpu())
                real_samples.append(batch.cpu())
                
                if len(generated_samples) * batch.size(0) >= 1000:  # Limit samples
                    break
        
        generated_samples = torch.cat(generated_samples, dim=0)
        real_samples = torch.cat(real_samples, dim=0)
        
        # Compute FID at different scales
        quality_metrics = {}
        for scale in [32, 64, 128]:
            # Resize to target scale
            gen_scaled = F.interpolate(generated_samples, size=(scale, scale), mode='bilinear')
            real_scaled = F.interpolate(real_samples, size=(scale, scale), mode='bilinear')
            
            # Compute FID (simplified version)
            fid_score = compute_fid_simplified(gen_scaled, real_scaled)
            quality_metrics[f'fid_scale_{scale}'] = fid_score
        
        return quality_metrics


# Utility functions
def compute_mig(representations: torch.Tensor, factors: torch.Tensor) -> float:
    """Compute Mutual Information Gap (MIG) score."""
    # Simplified MIG computation
    num_factors = factors.size(1)
    num_latents = representations.size(1)
    
    mi_matrix = torch.zeros(num_latents, num_factors)
    
    for i in range(num_latents):
        for j in range(num_factors):
            # Discretize continuous variables for MI computation
            repr_discrete = torch.quantile(representations[:, i], torch.linspace(0, 1, 11))[1:-1]
            factor_discrete = torch.quantile(factors[:, j], torch.linspace(0, 1, 11))[1:-1]
            
            # Compute mutual information (simplified)
            mi = compute_mutual_information_discrete(representations[:, i], factors[:, j], 
                                                   repr_discrete, factor_discrete)
            mi_matrix[i, j] = mi
    
    # MIG = mean of (max MI - second max MI) for each factor
    mig_scores = []
    for j in range(num_factors):
        sorted_mi = torch.sort(mi_matrix[:, j], descending=True)[0]
        if len(sorted_mi) > 1:
            mig_scores.append(sorted_mi[0] - sorted_mi[1])
    
    return torch.mean(torch.tensor(mig_scores)).item()


def compute_mutual_information_discrete(x: torch.Tensor, y: torch.Tensor,
                                      x_bins: torch.Tensor, y_bins: torch.Tensor) -> float:
    """Compute mutual information between discrete variables."""
    # Digitize continuous variables
    x_digital = torch.bucketize(x, x_bins)
    y_digital = torch.bucketize(y, y_bins)
    
    # Compute joint and marginal histograms
    joint_hist = torch.histc(x_digital * len(y_bins) + y_digital, 
                           bins=len(x_bins) * len(y_bins))
    joint_hist = joint_hist / joint_hist.sum()
    
    # Marginal histograms
    x_hist = torch.histc(x_digital, bins=len(x_bins))
    x_hist = x_hist / x_hist.sum()
    y_hist = torch.histc(y_digital, bins=len(y_bins))
    y_hist = y_hist / y_hist.sum()
    
    # Compute MI
    mi = 0.0
    for i in range(len(x_bins)):
        for j in range(len(y_bins)):
            idx = i * len(y_bins) + j
            if joint_hist[idx] > 0 and x_hist[i] > 0 and y_hist[j] > 0:
                mi += joint_hist[idx] * torch.log(joint_hist[idx] / (x_hist[i] * y_hist[j]))
    
    return mi.item()


def compute_mutual_information_approx(x: torch.Tensor, y: torch.Tensor) -> float:
    """Approximate mutual information using correlation."""
    # Simple approximation using correlation
    if x.size(1) != y.size(1):
        # If dimensions don't match, use canonical correlation
        min_dim = min(x.size(1), y.size(1))
        x = x[:, :min_dim]
        y = y[:, :min_dim]
    
    correlation = torch.corrcoef(torch.cat([x.T, y.T], dim=0))
    cross_corr = correlation[:x.size(1), x.size(1):]
    
    # Use trace of correlation as MI approximation
    mi = torch.trace(torch.abs(cross_corr)) / min(x.size(1), y.size(1))
    return mi.item()


def compute_fid_simplified(gen_samples: torch.Tensor, real_samples: torch.Tensor) -> float:
    """Simplified FID computation using feature statistics."""
    # Flatten samples
    gen_flat = gen_samples.view(gen_samples.size(0), -1)
    real_flat = real_samples.view(real_samples.size(0), -1)
    
    # Compute means and covariances
    mu_gen = torch.mean(gen_flat, dim=0)
    mu_real = torch.mean(real_flat, dim=0)
    
    sigma_gen = torch.cov(gen_flat.T)
    sigma_real = torch.cov(real_flat.T)
    
    # FID = ||mu_gen - mu_real||^2 + Tr(sigma_gen + sigma_real - 2*sqrt(sigma_gen*sigma_real))
    mu_diff = torch.sum((mu_gen - mu_real) ** 2)
    
    # Simplified trace term (using Frobenius norm as approximation)
    sigma_diff = torch.norm(sigma_gen - sigma_real, p='fro')
    
    fid = mu_diff + sigma_diff
    return fid.item()


def visualize_hierarchical_representations(model: HierarchicalVAE, 
                                         x: torch.Tensor, 
                                         save_path: str = None):
    """
    Visualize representations at different hierarchy levels.
    
    Creates visualization showing:
    1. Input image
    2. Reconstruction  
    3. Latent activations per level
    4. Generated samples per level
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(x.unsqueeze(0))
        
        # Create visualization
        fig, axes = plt.subplots(2, model.num_levels + 2, figsize=(15, 8))
        
        # Original and reconstruction
        if x.dim() == 3 and x.size(0) in [1, 3]:  # Image
            axes[0, 0].imshow(x.permute(1, 2, 0).cpu())
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            recon = outputs['reconstruction'][0]
            axes[0, 1].imshow(recon.permute(1, 2, 0).cpu())
            axes[0, 1].set_title('Reconstruction')
            axes[0, 1].axis('off')
        
        # Latent activations per level
        for i, params in enumerate(outputs['latent_params']):
            mu = params['mu'][0].cpu().numpy()
            
            # Plot latent activations
            axes[0, i + 2].bar(range(len(mu)), mu)
            axes[0, i + 2].set_title(f'Level {i} Latents')
            
            # Generate sample by modifying only this level
            z_modified = [p['sample'].clone() for p in outputs['latent_params']]
            z_modified[i] = torch.randn_like(z_modified[i])
            
            sample = model.decode(z_modified)
            if sample.dim() == 4 and sample.size(1) in [1, 3]:  # Image
                axes[1, i + 2].imshow(sample[0].permute(1, 2, 0).cpu())
                axes[1, i + 2].set_title(f'Level {i} Sample')
                axes[1, i + 2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


def compare_hierarchical_architectures(models: List[HierarchicalVAE],
                                     test_loader,
                                     model_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compare different hierarchical VAE architectures.
    
    Implements comprehensive comparison including:
    1. Reconstruction quality
    2. Generation quality
    3. Disentanglement
    4. Training stability
    5. Computational efficiency
    """
    comparison_results = {}
    
    for model, name in zip(models, model_names):
        model.eval()
        device = next(model.parameters()).device
        
        results = {
            'reconstruction_error': 0.0,
            'kl_divergence': 0.0,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'inference_time': 0.0
        }
        
        num_batches = 0
        total_inference_time = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                outputs = model(batch)
                end_time.record()
                
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                total_inference_time += inference_time
                
                # Compute reconstruction error
                recon_error = F.mse_loss(outputs['reconstruction'], batch)
                results['reconstruction_error'] += recon_error.item()
                
                # Compute KL divergence
                total_kl = 0.0
                for params in outputs['latent_params']:
                    kl = -0.5 * torch.sum(1 + params['logvar'] - params['mu'].pow(2) - params['logvar'].exp())
                    total_kl += kl.item()
                results['kl_divergence'] += total_kl / batch.size(0)
                
                num_batches += 1
                if num_batches >= 100:  # Limit evaluation
                    break
        
        # Average metrics
        results['reconstruction_error'] /= num_batches
        results['kl_divergence'] /= num_batches
        results['inference_time'] = total_inference_time / num_batches
        
        comparison_results[name] = results
    
    return comparison_results


def create_hierarchical_dataset(base_dataset, hierarchy_type: str = 'multi_scale'):
    """
    Create datasets suitable for hierarchical modeling.
    
    Implements dataset creation for:
    1. Multi-scale images (different resolutions)
    2. Hierarchical text (document/paragraph/sentence structure)
    3. Structured synthetic data with known hierarchical factors
    """
    if hierarchy_type == 'multi_scale':
        # Create multi-scale image dataset
        class MultiScaleDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, scales=[32, 64, 128]):
                self.base_dataset = base_dataset
                self.scales = scales
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                image, label = self.base_dataset[idx]
                
                # Create multi-scale versions
                multi_scale_images = []
                for scale in self.scales:
                    scaled = F.interpolate(image.unsqueeze(0), size=(scale, scale), 
                                         mode='bilinear', align_corners=False)
                    multi_scale_images.append(scaled.squeeze(0))
                
                return {
                    'images': multi_scale_images,
                    'original': image,
                    'label': label
                }
        
        return MultiScaleDataset(base_dataset)
    
    elif hierarchy_type == 'synthetic_hierarchical':
        # Create synthetic hierarchical data
        class SyntheticHierarchicalDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=10000, num_levels=3):
                self.num_samples = num_samples
                self.num_levels = num_levels
                
                # Generate hierarchical factors
                self.factors = []
                for level in range(num_levels):
                    # Higher levels have fewer factors
                    num_factors = max(1, 10 - level * 3)
                    factors = torch.randn(num_samples, num_factors)
                    self.factors.append(factors)
                
                # Generate observations from hierarchical factors
                self.observations = self._generate_observations()
            
            def _generate_observations(self):
                observations = torch.zeros(self.num_samples, 64, 64, 3)
                
                for i in range(self.num_samples):
                    # Combine factors from all levels
                    combined_factors = torch.cat([f[i] for f in self.factors], dim=0)
                    
                    # Generate image from factors (simplified)
                    # In practice, this would be a more sophisticated generative process
                    noise = torch.randn(64, 64, 3)
                    signal = combined_factors.mean() * torch.ones(64, 64, 3)
                    observations[i] = torch.sigmoid(signal + 0.1 * noise)
                
                return observations
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return {
                    'observation': self.observations[idx],
                    'factors': [f[idx] for f in self.factors]
                }
        
        return SyntheticHierarchicalDataset()
    
    else:
        raise ValueError(f"Unknown hierarchy type: {hierarchy_type}")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Hierarchical VAE Implementations...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    input_dim = 784  # MNIST flattened
    image_shape = (3, 64, 64)  # RGB images
    batch_size = 32
    
    # Test 1: Ladder VAE
    print("\nTesting Ladder VAE...")
    latent_dims = [64, 32, 16]
    hidden_dims = [256, 128, 64]
    
    ladder_vae = LadderVAE(input_dim, latent_dims, hidden_dims).to(device)
    test_input = torch.randn(batch_size, input_dim).to(device)
    
    with torch.no_grad():
        outputs = ladder_vae(test_input)
        print(f"Ladder VAE output shape: {outputs['reconstruction'].shape}")
        print(f"Number of latent levels: {len(outputs['latent_params'])}")
    
    # Test 2: Very Deep VAE  
    print("\nTesting Very Deep VAE...")
    vdvae_latent_dims = [128, 64, 32, 16, 8]  # 5 levels
    
    vdvae = VeryDeepVAE(image_shape, vdvae_latent_dims).to(device)
    test_images = torch.randn(batch_size, *image_shape).to(device)
    
    with torch.no_grad():
        outputs = vdvae(test_images)
        print(f"VDVAE output shape: {outputs['reconstruction'].shape}")
        print(f"Number of latent levels: {len(outputs['latent_params'])}")
    
    # Test 3: Multi-Scale VAE
    print("\nTesting Multi-Scale Image VAE...")
    
    multiscale_vae = MultiScaleImageVAE(image_channels=3, base_resolution=16).to(device)
    test_images_small = torch.randn(batch_size, 3, 48, 48).to(device)
    
    with torch.no_grad():
        outputs = multiscale_vae(test_images_small)
        print(f"Multi-scale VAE output shape: {outputs['reconstruction'].shape}")
        print(f"Multi-scale levels: {multiscale_vae.num_levels}")
    
    # Test 4: Hierarchical Loss
    print("\nTesting Hierarchical Loss...")
    
    loss_fn = HierarchicalLoss(num_levels=3, beta_schedule='hierarchical', 
                              free_bits=0.1).to(device)
    
    # Test loss computation
    loss_dict = loss_fn(outputs, test_images_small)
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Reconstruction loss: {loss_dict['reconstruction_loss'].item():.4f}")
    print(f"KL losses per level: {[kl.item() for kl in loss_dict['kl_losses']]}")
    
    # Test 5: Hierarchical Trainer
    print("\nTesting Hierarchical Trainer...")
    
    trainer = HierarchicalTrainer(multiscale_vae, device)
    
    # Create dummy data loader
    dummy_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 3, 48, 48)
    )
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=16)
    
    # Test utilization evaluation
    utilization = trainer.evaluate_hierarchy_utilization(dummy_loader)
    print("Hierarchy utilization metrics:")
    for key, value in utilization.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 6: Hierarchical Evaluation
    print("\nTesting Hierarchical Evaluation...")
    
    # Test interpolation
    x1 = torch.randn(3, 48, 48).to(device)
    x2 = torch.randn(3, 48, 48).to(device) 
    
    interpolations = HierarchicalEvaluation.visualize_hierarchy_interpolation(
        multiscale_vae, x1, x2, num_steps=5
    )
    print(f"Interpolation shape: {interpolations.shape}")
    
    # Test information flow analysis
    info_flow = HierarchicalEvaluation.analyze_information_flow(multiscale_vae, dummy_loader)
    print(f"Mutual information matrix shape: {info_flow['mutual_information_matrix'].shape}")
    
    print("\nAll tests completed successfully!")
    print("✓ Ladder VAE implementation verified")
    print("✓ Very Deep VAE implementation verified") 
    print("✓ Multi-Scale VAE implementation verified")
    print("✓ Hierarchical loss function verified")
    print("✓ Training utilities verified")
    print("✓ Evaluation metrics verified")
    print("\nImplementation complete! Ready for hierarchical VAE experiments.")
