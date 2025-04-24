import numpy as np


def float_to_q1_8(arr):
    """
    Converts a numpy array of floats to Q1.8 fixed-point binary representation.
    Q1.8: 1 sign bit, 7 integer bits, 8 fractional bits
    Range: -128 to 127.99609375 (clamped)
    """
    arr = np.clip(arr, -128, 127.99609375)
    scaled = np.round(arr * 256).astype(np.int16)  # 16-bit signed integer
    return scaled

def q1_8_to_float(q_arr):
    """
    Converts Q1.8 fixed-point binary back to float.
    """
    return q_arr.astype(np.float32) / 256.0

def embedding_to_binary_file(embedding, filename):
    """
    Converts an embedding to Q1.8 and saves as binary file.
    """
    q_embedding = float_to_q1_8(embedding)
    q_embedding.tofile(filename)

def binary_file_to_embedding(filename, shape):
    """
    Reads a binary file and restores the embedding as float.
    """
    q_embedding = np.fromfile(filename, dtype=np.int16).reshape(shape)
    return q1_8_to_float(q_embedding)

# --- Secure Enrollment: XOR encryption with random bit-vector R ---

def generate_random_bitvector(shape, bit_depth=16):
    """
    Generate a random bit-vector R of the same shape as the template.
    """
    max_val = 2**bit_depth - 1
    R = np.random.randint(0, max_val + 1, size=shape, dtype=np.uint16)
    return R

def xor_encrypt(template, R):
    """
    Encrypt or decrypt template using XOR with random bit-vector R.
    """
    return np.bitwise_xor(template, R)

def secure_enrollment(Xfb, Nb):
    """
    Encrypt templates Xfb and Nb with random pad R.
    Returns encrypted templates and R.
    """
    assert Xfb.shape == Nb.shape, "Templates must have the same shape"
    R = generate_random_bitvector(Xfb.shape, bit_depth=16)
    X_enc = xor_encrypt(Xfb, R)
    N_enc = xor_encrypt(Nb, R)
    return X_enc, N_enc, R

def secure_decrypt(X_enc, N_enc, R):
    """
    Decrypt encrypted templates using XOR with pad R.
    """
    Xfb = xor_encrypt(X_enc, R)
    Nb = xor_encrypt(N_enc, R)
    return Xfb, Nb

# --- Octet Generation and Hamming Distance Analysis ---

def generate_octets(noise_code):
    """
    Splits a binary noise code (numpy array of 0s and 1s) into 4-bit octets.
    """
    assert len(noise_code) % 4 == 0, "Noise code length must be multiple of 4"
    return noise_code.reshape((-1, 4))

def hamming_distance(a, b):
    """
    Computes total Hamming distance between two binary arrays.
    """
    assert a.shape == b.shape, "Arrays must be same shape"
    return np.sum(a != b)

def octet_hamming_distance(octets1, octets2):
    """
    Computes average bit difference per 4-bit octet and normalized Hamming distance.
    """
    assert octets1.shape == octets2.shape, "Octet arrays must be same shape"
    diff = octets1 != octets2
    bit_diff_per_octet = np.sum(diff, axis=1)
    avg_bits_diff = np.mean(bit_diff_per_octet)
    normalized_hd = avg_bits_diff / 4.0
    return avg_bits_diff, normalized_hd

# --- Noise Code Generation for Testing Octet Behavior ---

def generate_noise_code(length=128, flip_prob=0.25):
    """
    Generate a random binary noise code and a noisy version simulating mated pair.
    """
    base_code = np.random.randint(0, 2, size=length, dtype=np.uint8)
    noise = np.random.rand(length) < flip_prob
    noisy_code = np.bitwise_xor(base_code, noise.astype(np.uint8))
    return base_code, noisy_code

# --- Example Workflow Demonstration ---

def example_workflow():
    print("=== Step 1: Generate example float embedding ===")
    # Example float embedding (simulate with random floats)
    image_embedding = np.random.uniform(-1, 1, size=128).astype(np.float32) * 50  # scaled floats
    
    print("Original float embedding (first 10 values):", image_embedding[:10])
    
    print("\n=== Step 2: Convert float embedding to Q1.8 fixed-point ===")
    Xfb = float_to_q1_8(image_embedding)
    print("Fixed-point embedding (first 10 values):", Xfb[:10])
    
    print("\n=== Step 3: Simulate another template Nb (e.g., noise or another embedding) ===")
    Nb = float_to_q1_8(image_embedding + np.random.normal(0, 0.1, size=image_embedding.shape))
    print("Nb fixed-point embedding (first 10 values):", Nb[:10])
    
    print("\n=== Step 4: Secure Enrollment (encrypt templates) ===")
    X_enc, N_enc, R = secure_enrollment(Xfb.astype(np.uint16), Nb.astype(np.uint16))
    print("Encrypted Xfb (first 10 values):", X_enc[:10])
    print("Encrypted Nb (first 10 values):", N_enc[:10])
    
    print("\n=== Step 5: Decrypt templates to verify correctness ===")
    Xfb_dec, Nb_dec = secure_decrypt(X_enc, N_enc, R)
    print("Decrypted Xfb matches original:", np.array_equal(Xfb_dec, Xfb.astype(np.uint16)))
    print("Decrypted Nb matches original:", np.array_equal(Nb_dec, Nb.astype(np.uint16)))
    
    print("\n=== Step 6: Generate binary noise codes from fixed-point embeddings ===")
    # Convert fixed-point embeddings to binary arrays (bit-level)
    def int_to_bits(arr, bit_width=16):
        return np.unpackbits(arr.view(np.uint8)).reshape(-1, bit_width)[:, ::-1]  # MSB last
    
    # For simplicity, use only lower 128 bits (first 8 elements * 16 bits = 128 bits)
    bits_Xfb = int_to_bits(Xfb[:8])
    bits_Nb = int_to_bits(Nb[:8])
    
    # Flatten to 1D binary arrays
    noise_code_X = bits_Xfb.flatten()
    noise_code_N = bits_Nb.flatten()
    
    print("Noise code length:", len(noise_code_X))
    
    print("\n=== Step 7: Generate octets and analyze Hamming distance ===")
    octets_X = generate_octets(noise_code_X)
    octets_N = generate_octets(noise_code_N)
    
    avg_bits_diff, norm_hd = octet_hamming_distance(octets_X, octets_N)
    print(f"Average bits difference per octet: {avg_bits_diff:.2f} bits (~{norm_hd*100:.1f}%)")
    
    print("\n=== Step 8: Test with mated and non-mated pairs using noise code generation ===")
    base_code, mated_code = generate_noise_code(length=128, flip_prob=0.25)
    non_mated_code = np.random.randint(0, 2, size=128, dtype=np.uint8)
    
    base_octets = generate_octets(base_code)
    mated_octets = generate_octets(mated_code)
    non_mated_octets = generate_octets(non_mated_code)
    
    mated_bits_diff, mated_norm_hd = octet_hamming_distance(base_octets, mated_octets)
    non_mated_bits_diff, non_mated_norm_hd = octet_hamming_distance(base_octets, non_mated_octets)
    
    print(f"Mated pair average bits difference per octet: {mated_bits_diff:.2f} bits (~{mated_norm_hd*100:.1f}%)")
    print(f"Non-mated pair average bits difference per octet: {non_mated_bits_diff:.2f} bits (~{non_mated_norm_hd*100:.1f}%)")

if __name__ == "__main__":
    example_workflow()
