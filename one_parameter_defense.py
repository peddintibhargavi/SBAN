import time
import numpy as np
def one_parameter_defense(v, m=5, e=0.1, max_attempts=10, seed=None):
    """
    Privacy-preserving feature transformation with guaranteed low noise-identity correlation.
    
    Args:
        v: Input feature vector (normalized)
        m: Magnitude parameter (typically 3-10)
        e: Epsilon for numerical stability
        max_attempts: Maximum tries to achieve low correlation
        seed: Optional seed for deterministic output
    
    Returns:
        Protected feature vector with same dimension as input
    """
    v = np.array(v, dtype=np.float32)
    if len(v.shape) != 1:
        raise ValueError("Input must be 1D vector")
    
    # Normalize input
    norm = np.linalg.norm(v)
    if abs(norm - 1.0) > 1e-5:
        v = v / (norm + e)
    
    # Generate properly decorrelated noise
    best_noise = None
    lowest_corr = float('inf')
    
    # Use provided seed or generate one from current time
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
    
    # Fixed seed for testing - use the same seed for all tests to ensure determinism
    # seed = 12345
    
    rng = np.random.RandomState(seed)
    
    for _ in range(max_attempts):
        noise = rng.randn(v.shape[0])
        
        # Make noise orthogonal to identity vector
        noise = noise - np.dot(noise, v) * v
        noise_norm = np.linalg.norm(noise)
        
        if noise_norm > 1e-5:
            noise = noise / noise_norm
            current_corr = abs(np.dot(v, noise))
            
            if current_corr < 0.1:  # Our target threshold
                best_noise = noise
                break
            elif current_corr < lowest_corr:
                lowest_corr = current_corr
                best_noise = noise
    
    if best_noise is None:
        raise ValueError("Could not generate sufficiently decorrelated noise")
    
    # Apply perturbation
    protected_v = v + m * best_noise
    
    # Final normalization
    protected_v = protected_v / (np.linalg.norm(protected_v) + e)
    
    return protected_v