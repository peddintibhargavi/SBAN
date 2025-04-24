import numpy as np
from one_parameter_defense import one_parameter_defense

def float_to_q18(x: np.ndarray) -> np.ndarray:
    """Convert float array to Q1.8 fixed-point representation (1 sign bit, 8 fractional bits)."""
    return np.clip((x * 256).astype(np.int16), -32768, 32767)  # Q1.8 range: [-256, 255.996]

def cbe_random(input_dim, output_dim=512, r=3):
    """
    Generate random projection matrices for Circulant Binary Embedding with Q1.8 support.
    """
    rng = np.random.RandomState(1234)
    R = rng.randn(r, input_dim)
    for i in range(r):
        R[i] = float_to_q18(R[i] / np.linalg.norm(R[i])) / 256.0
    
    D = rng.choice([-1, 1], (r, output_dim))
    return {"R": R, "D": D, "input_dim": input_dim, "output_dim": output_dim, "r": r}

def cbe_prediction(cbe_model, x):
    """Q1.8 compatible prediction"""
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    
    if x.shape[1] != cbe_model["input_dim"]:
        raise ValueError(f"Input dimension mismatch. Expected {cbe_model['input_dim']}, got {x.shape[1]}")
    
    binary_codes = np.zeros((x.shape[0], cbe_model["output_dim"]), dtype=np.int32)
    
    for i in range(x.shape[0]):
        z = np.zeros(cbe_model["output_dim"])
        for j in range(cbe_model["r"]):
            proj = np.dot(float_to_q18(x[i]), float_to_q18(cbe_model["R"][j])) / 65536.0  # Q1.8 multiplication
            signed_proj = proj * cbe_model["D"][j]
            z += signed_proj
        
        binary_codes[i] = (z >= 0).astype(np.int32)
    
    return binary_codes[0] if binary_codes.shape[0] == 1 else binary_codes

def cbe_prediction_with_opd(model, X):
    """Protected prediction pipeline"""
    X_d = one_parameter_defense(X)
    return cbe_prediction(model, X_d)
if __name__ == "__main__":
    model = cbe_random(512)
    V = []
    for i in range(100):
        A = np.random.randn(512)
        A /= np.linalg.norm(A)
        B1 = one_parameter_defense(A)
        B2 = one_parameter_defense(A)
        v1, R1 = cbe_prediction(model, B1) #Save random vectors
        v2, R2 = cbe_prediction(model, B2) # Save random vectors
        hd = 1 - np.sum(np.logical_xor(v1, v2)) / B1.shape[0]
        V.append(abs(B1 @ B2 - hd))
        print(np.average(V) * 100, np.max(V) * 100, np.min(V) * 100)
    import matplotlib.pyplot as plt
    plt.hist(V, 3)
    plt.show()
