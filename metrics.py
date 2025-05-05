import numpy as np

import jax.numpy as jnp 
from jax import jit 

from scipy.linalg import sqrtm

from jax.scipy import linalg


def sqrtm_jax(matrix, eps=1e-10):
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    
    sqrt_eigvals = jnp.sqrt(jnp.maximum(eigvals, eps))
    
    return eigvecs @ jnp.diag(sqrt_eigvals) @ eigvecs.T


@jit
def wasserstein_gaussian(samples, target_mean, target_cov):
    # NOTE: CPU-bound due to sqrtm 
    sample_mean = jnp.mean(samples, axis=0)
    sample_cov = jnp.cov(samples, rowvar=False)
    
    mean_diff = sample_mean - target_mean
    mean_term = jnp.linalg.norm(mean_diff) ** 2
    
    sqrt_target_cov = linalg.sqrtm(target_cov)
    sqrt_sample_cov = linalg.sqrtm(sample_cov) 

    return jnp.sqrt(mean_term + jnp.linalg.norm(sqrt_sample_cov - sqrt_target_cov, 'fro')** 2)