import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd


# Diffusion Process
class GaussianDiffusion:
    """
    Implements the forward diffusion process from DDPM:  
      q(x_t|x_0) = N(α(t) * x_0, σ(t)^2 * I)
    where:
      x_t = α(t) * x_0 + σ(t) * ε,  with ε ~ N(0, I)      
    and the functions α(t) (reflects the total amount of the original signal that remains at each timestep t) 
    and σ(t) (reflects how much noises is added at each timestep t)  are defined via:
      α(t) = sqrt(ᾱ_t) and σ(t) = sqrt(1 - ᾱ_t)
    with:
      ᾱ_t = ∏_{i=1}^{t} (1 - β_i)
    """
    def __init__(
        self,
        num_diffusion_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        device: torch.device = torch.device("cpu")
    ):
        self._num_diffusion_timesteps = num_diffusion_timesteps
        self.device = device
        # Betas
        self._betas = self._get_beta_schedule(
            beta_schedule, beta_start, beta_end, num_diffusion_timesteps
        )
        # Alphas
        alphas_bar = np.cumprod(1.0 - self._betas)
        alphas_bar = np.concatenate(([1.], alphas_bar))  # Makes it convenient for generating sample at t=0 (no noise added)
        self._alpha_bars = torch.tensor(alphas_bar, dtype=torch.float32, device=self.device)
        # Precompute α(t)=sqrt(ᾱ_t) and σ(t)=sqrt(1 - ᾱ_t) as torch tensors.
        self._alphas = torch.tensor(np.sqrt(alphas_bar), dtype=torch.float32, device=self.device)
        self._sigmas = torch.tensor(np.sqrt(1 - alphas_bar), dtype=torch.float32, device=self.device)

    @staticmethod
    def _get_beta_schedule(beta_schedule: str, beta_start: float, beta_end: float, num_diffusion_timesteps: int) -> np.ndarray:
        """
        Returns a numpy array of beta values given the schedule type.
        Supported schedules: "quad", "linear", "const", "jsd", and "sigmoid".
        """
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)        
        if beta_schedule == "quad":
            betas = (np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64
            ) ** 2)
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":
            betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_diffusion_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")
        
        assert betas.shape == (num_diffusion_timesteps,)
        return betas

    @property
    def tmin(self):
        return 1

    @property
    def tmax(self):
        return self._num_diffusion_timesteps

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Returns α(t) = sqrt(ᾱ_t) for the given timesteps."""
        return self._alphas[t.long()]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Returns σ(t) = sqrt(1 - ᾱ_t) for the given timesteps."""
        return self._sigmas[t.long()]

    def sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """ Draws samples from the forward diffusion process q(x_t|x_0):
          x_t = α(t) * x0 + σ(t) * ε,
        where ε ~ N(0, I).
        """
        alpha_t = self.alpha(t).view(-1, 1) 
        sigma_t = self.sigma(t).view(-1, 1)
        return alpha_t * x0 + sigma_t * eps


# Time Sampler
class TimeSampler:
    def __init__(self, t_min: int, t_max: int):
        """
        Initializes the TimeSampler with the minimum and maximum timestep.
        Args:
            t_min (int): The minimum timestep (inclusive).
            t_max (int): The maximum timestep (exclusive).
        """
        self.t_min = t_min
        self.t_max = t_max

    def sample(self, size: int, strategy: str = "antithetic") -> torch.Tensor:
        """
        Samples timesteps according to the specified strategy.
        Args:
            size (int): The number of timesteps to generate.
            strategy (str): Sampling strategy; either "antithetic" (default) or "uniform".
        Returns:
            torch.Tensor: A tensor of sampled timesteps.
        """
        if strategy == "uniform":
            # Simply sample uniformly between t_min and t_max.
            return torch.randint(low=self.t_min, high=self.t_max, size=(size,))
        elif strategy == "antithetic":
            # Calculate the number of samples needed from one side (round up if odd)
            half_n = size // 2 + (size % 2)
            # Sample half_n timesteps uniformly.
            t_half = torch.randint(low=self.t_min, high=self.t_max, size=(half_n,))
            # Compute antithetic counterparts.
            t_antithetic = self.t_max - t_half - 1
            # Concatenate both and slice to exactly 'size' samples.
            t_full = torch.cat([t_half, t_antithetic], dim=0)[:size]
            return t_full
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'uniform' or 'antithetic'.")


# Neural Network to predict the added noise in the forward process
class NoisePredictor(nn.Module):
    def __init__(self, data_dim, time_emb_dim, label_emb_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, total_timesteps: int = 1000):
        """
        Args:
            data_dim (int): Dimension of the data
            time_emb_dim (int): Dimension of the time embedding. If set to 1, a simple normalized timestep is used. 
                                Otherwise, we apply sinusoidal embedding as in the standard DDPM implementation
            label_emb_dim (int): Dimension for projecting the continuous label.
            hidden_dim_1, hidden_dim_2, hidden_dim_3 (int)
            total_timesteps (int): Total number of diffusion timesteps.
        """
        super(NoisePredictor, self).__init__()
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim
        self.label_emb_dim = label_emb_dim
        self.total_timesteps = total_timesteps
        
        # Projection for the continuous label.
        self.label_proj = nn.Linear(1, label_emb_dim)
        
        # The input dimension is the sum of the data dimension, the time embedding dimension, and the label embedding dimension.
        self.input_dim = data_dim + time_emb_dim + label_emb_dim
        # First layer: from input_dim to hidden_dim_1.
        self.fc1 = nn.Linear(self.input_dim, hidden_dim_1)
        self.norm1 = nn.LayerNorm(hidden_dim_1)
        # Second layer: before this layer, we inject the time and label embeddings again.
        # So its input dimension is hidden_dim_1 + time_emb_dim + label_emb_dim.
        self.fc2 = nn.Linear(hidden_dim_1 + time_emb_dim + label_emb_dim, hidden_dim_2)
        self.norm2 = nn.LayerNorm(hidden_dim_2)
        # Third layer: maps from hidden_dim_2 to hidden_dim_3.
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.norm3 = nn.LayerNorm(hidden_dim_3)
        # Final output layer: maps from hidden_dim_3 to data_dim.
        self.out = nn.Linear(hidden_dim_3, data_dim)
        # A projection Layer from input_dim to data_dim so that we can add the residual connection
        self.res_proj = nn.Linear(self.input_dim, data_dim)
        
        self.activation = nn.ReLU()

    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """
        Generate a time embedding for each timestep.
        If embedding_dim == 1, returns a normalized timestep (t / total_timesteps).
        Otherwise, returns a sinusoidal embedding as in the DDPM paper.
        
        Args:
            timesteps (torch.Tensor): 1D tensor of timesteps (shape: [batch_size]).
            embedding_dim (int): The desired dimension of the embedding.
        Returns:
            torch.Tensor: A tensor of shape [batch_size, embedding_dim] representing the time embeddings.
        """
        assert timesteps.ndim == 1, "timesteps should be a 1D tensor"

        if embedding_dim == 1:
            # Normalize the timestep value to [0, 1].
            return timesteps.float()[:, None] / self.total_timesteps
        else:
            half_dim = embedding_dim // 2
            emb_factor = math.log(10000) / (half_dim - 1)
            # Compute an exponentially decreasing set of factors.
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb_factor)
            # Multiply each timestep by these factors.
            emb = timesteps.float()[:, None] * emb[None, :]
            # Compute sine and cosine components.
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            if embedding_dim % 2 == 1:
                # If embedding_dim is odd, pad with an extra zero.
                emb = F.pad(emb, (0, 1))
            return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Noisy data of shape [batch_size, data_dim].
            t (torch.Tensor): Timestep tensor of shape [batch_size].
            label (torch.Tensor): Continuous label of shape [batch_size] or [batch_size, 1].       
        Returns:
            torch.Tensor: Predicted noise of shape [batch_size, data_dim].
        """
        # Get time embedding.
        t_emb = self.get_timestep_embedding(t, self.time_emb_dim)
        # Ensure label has shape [batch_size, 1].
        if label.ndim == 1:
            label = label.unsqueeze(1)
        label_emb = self.label_proj(label) # [batch_size, label_emb_dim]
        x_in = torch.cat([x, t_emb, label_emb], dim=1) # Concatenate the noisy data, time embedding, and label embedding [batch_size, input_dim].
        h = self.activation(self.norm1(self.fc1(x_in)))
        h = torch.cat([h, t_emb, label_emb], dim=1) # Inject the time and the label condition in the intermediate layer
        h = self.activation(self.norm2(self.fc2(h)))
        h = self.activation(self.norm3(self.fc3(h)))
        out = self.out(h) # [batch_size, data_dim]
        res = self.res_proj(x_in)
        
        return out + res #residual


# Broadcasting Function
def bcast_right(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Util function for broadcasting to the right."""
    if x.ndim > ndim:
        raise ValueError(f'Cannot broadcast a value with {x.ndim} dims to {ndim} dims.')
    elif x.ndim < ndim:
        difference = ndim - x.ndim
        return x.view(x.shape + (1,) * difference)
    else:
        return x


# The whole generative modeling process (forwad diffusion + backward denoising) 
class DDPM(nn.Module):
    """The forward diffusion + the backward denoising"""

    def __init__(self, snp_dim, diffusion_process, time_sampler, noise_predictor):
        super(DDPM, self).__init__()
        self._snp_dim = snp_dim
        self._diffusion = diffusion_process
        self._time_sampler = time_sampler
        self._noise_predictor = noise_predictor

    def loss(self, x0: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Computes MSE between the true noise and predicted noise"""
        t = self._time_sampler.sample(size=x0.shape[0]).to(x0.device)  # sample timesteps
        eps = torch.randn_like(x0, device=x0.device)  # sample noise
        xt = self._diffusion.sample(x0, t, eps).to(x0.device)  # generate a noisy data using the forward diffusion process
        predicted_noise = self._noise_predictor(xt, t, label)  # predict added noise
        loss = torch.mean((predicted_noise - eps) ** 2)  # compute MSE loss between predicted and true noise
        return loss

    def one_reverse_step(self, xt: torch.Tensor, t: int, label: torch.Tensor) -> torch.Tensor:
        """
        Computes one step of the denoising process.
        This function computes a sample from the Gaussian distribution  p(x_{t-1} | x_t, x0_pred) based on the current noisy image xt, the 
        current timestep t, and a conditioning label.
        Args:
            xt (torch.Tensor): The current noisy input (x_t).
            t (int): The current timestep (should be >= 1).
            label (torch.Tensor): The conditioning label.
        Returns:
            torch.Tensor: A sample from p(x_{t-1}| x_t, x0_pred), i.e. the denoised image.
        """
        t = t * torch.ones((xt.shape[0],), dtype=torch.int32, device=xt.device)
        eps_pred = self._noise_predictor(xt, t, label)
        sqrt_a_t = self._diffusion.alpha(t) / self._diffusion.alpha(t - 1)
        inv_sqrt_a_t = bcast_right(1.0 / sqrt_a_t, xt.ndim) # the scaling factor of the noisy image from t to t-1
        beta_t = 1.0 - sqrt_a_t ** 2
        beta_t = bcast_right(beta_t, xt.ndim) # variance 
        inv_sigma_t = bcast_right(1.0 / self._diffusion.sigma(t), xt.ndim) # for scaling the predicted noise
        mean = inv_sqrt_a_t * (xt - beta_t * inv_sigma_t * eps_pred) # mean of the reverse process
        std = torch.sqrt(beta_t)
        z = torch.randn_like(xt)

        return mean + std * z

    def sample(self, sample_size: int, label: torch.Tensor) -> torch.Tensor:
        """
        Generates samples from the reverse diffusion process starting from x_T (pure noise).
        Args:
            sample_size (int): The number of samples to generate.
            label (torch.Tensor): The conditioning label.
        Returns:
            torch.Tensor: Generated samples from the model.
        """
        with torch.no_grad():
            x = torch.randn(sample_size, self._snp_dim, device=label.device)
            # Iterate over timesteps from T down to 1.
            for t in range(self._diffusion.tmax, 0, -1):
                x = self.one_reverse_step(x, t, label)
        return x

    def generate(self, pheno_tensor: torch.Tensor) -> np.ndarray:
        """
        Reconstruction from PCA space to the original data space
        Args:
            pheno_tensor (torch.Tensor): The conditioning phenotype tensor
        Returns:
            np.ndarray: The 2D array for reconstructed genotype matrix
        """
        all_reconstructed_data = np.empty((pheno_tensor.shape[0], 0))
        dims = [250,228,214,224,206,204,189,192,181,186,196,184,155,155,175,145,162,166,153,118,139,141,122,132,121,113,120,122,126] # The PCA dimension for each chromosome
        cumdims = np.cumsum(dims)
        start_indices = np.insert(cumdims, 0, 0)[:-1] # Prepend a 0 at the beginning to get the start indices.
        # Generation of PCA latent representation
        self.eval()
        with torch.no_grad():
            AG_PCA = np.array(self.sample(pheno_tensor.shape[0], pheno_tensor).cpu())
        AG_PCA_slices = [AG_PCA[:, start:stop] for start, stop in zip(start_indices, cumdims)]
        for chr in range(1,30):
            result_folder = "../../../../pca/pca_data/cows/all/ch"+str(chr) # The folder where the pca components and the pca mean are stored
            pca_components = np.load(result_folder + "/pca_components.npy")
            pca_mean = np.load(result_folder + "/pca_mean.npy")
            reconstructed_data = np.dot(AG_PCA_slices[chr-1], pca_components) + pca_mean
            reconstructed_data = np.rint(reconstructed_data).clip(0, 2)  
            all_reconstructed_data = np.concatenate((all_reconstructed_data, reconstructed_data), axis=1)
        
        return all_reconstructed_data