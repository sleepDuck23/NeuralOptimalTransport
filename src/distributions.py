import torch
import numpy as np
import random
from scipy.linalg import sqrtm
from sklearn import datasets

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)
            
        return batch[:size].to(self.device)
    
    
class SwissRollSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda'
    ):
        super(SwissRollSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        
    def sample(self, batch_size=10):
        batch = datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype('float32')[:, [0, 2]] / 7.5
        return torch.tensor(batch, device=self.device)
    
class StandardNormalSampler(Sampler):
    def __init__(self, dim=1, device='cuda'):
        super(StandardNormalSampler, self).__init__(device=device)
        self.dim = dim
        
    def sample(self, batch_size=10):
        return torch.randn(batch_size, self.dim, device=self.device)
    
    
class Mix8GaussiansSampler(Sampler):
    def __init__(self, with_central=False, std=1, r=12, dim=2, device='cuda'):
        super(Mix8GaussiansSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        self.std, self.r = std, r
        
        self.with_central = with_central
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        if self.with_central:
            centers.append((0, 0))
        self.centers = torch.tensor(centers, device=self.device, dtype=torch.float32)
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.randn(batch_size, self.dim, device=self.device)
            indices = random.choices(range(len(self.centers)), k=batch_size)
            batch *= self.std
            batch += self.r * self.centers[indices, :]
        return batch
    

class MixNGaussiansSampler(Sampler):
    def __init__(self, n=5, dim=2, std=1, step=9, device='cuda'):
        super(MixNGaussiansSampler, self).__init__(device=device)
        
        assert dim == 1
        self.dim = 1
        self.std, self.step = std, step
        
        self.n = n
        
        grid_1d = np.linspace(-(n-1) / 2., (n-1) / 2., n)
        self.centers = torch.tensor(grid_1d, device=self.device,)
        
    def sample(self, batch_size=10):
        batch = torch.randn(batch_size, self.dim, device=self.device)
        indices = random.choices(range(len(self.centers)), k=batch_size)
        with torch.no_grad():
            batch *= self.std
            batch += self.step * self.centers[indices, None]
        return batch
    

class Transformer(object):
    def __init__(self, device='cuda'):
        self.device = device
        

class StandardNormalScaler(Transformer):
    def __init__(self, base_sampler, batch_size=1000, device='cuda'):
        super(StandardNormalScaler, self).__init__(device=device)
        self.base_sampler = base_sampler
        batch = self.base_sampler.sample(batch_size).cpu().detach().numpy()
        
        mean, cov = np.mean(batch, axis=0), np.matrix(np.cov(batch.T))
        
        self.mean = torch.tensor(
            mean, device=self.device, dtype=torch.float32
        )
        
        multiplier = sqrtm(cov)
        self.multiplier = torch.tensor(
            multiplier, device=self.device, dtype=torch.float32
        )
        self.inv_multiplier = torch.tensor(
            np.linalg.inv(multiplier),
            device=self.device, dtype=torch.float32
        )
        torch.cuda.empty_cache()
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.tensor(self.base_sampler.sample(batch_size), device=self.device)
            batch -= self.mean
            batch @= self.inv_multiplier
        return batch
    
# ---------------- 1. 5D Gaussian Mixture Sampler ----------------
class Mix5DGaussiansSampler(Sampler):
    def __init__(self, num_components=5, std=1, r=10, dim=5, device='cuda'):
        super(Mix5DGaussiansSampler, self).__init__(device=device)
        assert dim == 5  # Ensure it's a 5D distribution
        self.dim = dim
        self.std, self.r = std, r
        self.num_components = num_components
        
        # Randomly initialize Gaussian component centers
        self.centers = torch.randn(num_components, dim, device=self.device) * r

    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.randn(batch_size, self.dim, device=self.device) * self.std
            indices = torch.randint(0, self.num_components, (batch_size,), device=self.device)
            batch += self.centers[indices]  # Shift by selected Gaussian component center
        return batch

# ---------------- 2. 5D Heavy-Tailed Distribution (Student's t-distribution) ----------------
class HeavyTailed5DSampler(Sampler):
    def __init__(self, df=2, dim=5, device='cuda'):
        """Heavy-tailed distribution using Student's t-distribution"""
        super(HeavyTailed5DSampler, self).__init__(device=device)
        self.df = df  # Degrees of freedom
        self.dim = dim

    def sample(self, batch_size=10):
        """Generate samples from a 5D heavy-tailed Student's t-distribution"""
        with torch.no_grad():
            batch = torch.from_numpy(np.random.standard_t(self.df, size=(batch_size, self.dim))).float().to(self.device)
        return batch

# ---------------- 3. 5D Data on a Low-Dimensional Manifold (Swiss Roll in 5D) ----------------
class SwissRoll5DSampler(Sampler):
    def __init__(self, noise=0.5, dim=5, device='cuda'):
        """Swiss roll data with intrinsic 2D structure embedded in 5D"""
        super(SwissRoll5DSampler, self).__init__(device=device)
        assert dim == 5  # Ensure it’s a 5D manifold
        self.noise = noise
        self.dim = dim

    def sample(self, batch_size=10):
        """Generate Swiss roll data and embed in 5D"""
        with torch.no_grad():
            t = 3 * np.pi * (1 + 2 * np.random.rand(batch_size))  # 2D Swiss roll intrinsic structure
            x = t * np.cos(t)
            y = t * np.sin(t)
            z = 5 * np.random.rand(batch_size)  # Adding depth in 3D
            extra_dims = np.random.randn(batch_size, 2) * self.noise  # Extra dimensions with noise
            
            batch = np.column_stack((x, y, z, extra_dims)) / 10  # Normalize
            return torch.tensor(batch, dtype=torch.float32, device=self.device)