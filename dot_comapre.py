# Imports from previous steps
from feature_conv import float_to_q1_8, q1_8_to_float
from feature_conv import secure_enrollment, secure_decrypt
from feature_conv import generate_octets

import numpy as np

def distribute_template_shares(X_enc, Y_enc, M_enc, N_enc):
    """
    Distributes the encrypted templates and masks between two parties.
    """
    P1 = {
        'X': X_enc,
        'M': M_enc
    }
    
    P2 = {
        'Y': Y_enc,
        'N': N_enc
    }
    
    return P1, P2
def secure_and_masked_dot_product(P1, P2, octets_X, octets_Y, octets_M, octets_N):
    """
    Secure dot product computation using masked AND operations.
    Each AND operation is masked by the corresponding octet.
    """
    assert octets_X.shape == octets_Y.shape == octets_M.shape == octets_N.shape
    num_octets = octets_X.shape[0]
    dot_product = 0

    # For each octet, perform a masked AND and sum the result
    for i in range(num_octets):
        # Masked AND for this octet (bitwise AND, then mask with octet_M and octet_N)
        # This is a simplified simulation: in a real protocol, this would be performed as a secure two-party computation (e.g., garbled circuits or oblivious transfer)
        and_result = np.bitwise_and(octets_X[i], octets_Y[i])
        masked_and = np.bitwise_and(and_result, octets_M[i])
        masked_and = np.bitwise_and(masked_and, octets_N[i])
        dot_product += np.sum(masked_and)
    return dot_product

def compute_hamming_distance_bits(octets1, octets2):
    """
    Compute average bits difference per octet and normalized Hamming distance.
    """
    assert octets1.shape == octets2.shape, "Octet arrays must be same shape"
    diff = octets1 != octets2
    bit_diff_per_octet = np.sum(diff, axis=1)
    avg_bits_diff = np.mean(bit_diff_per_octet)
    normalized_hd = avg_bits_diff / 4.0
    return avg_bits_diff, normalized_hd

def apply_correction(dot_product_score, observed_hd, expected_hd=0.25):
    """
    Apply a correction mechanism if noise mismatch deviates from expected 25% Hamming distance.
    This is a simple linear correction for demonstration.
    """
    # If observed HD deviates significantly, scale the score accordingly
    correction_factor = expected_hd / observed_hd if observed_hd != 0 else 1.0
    corrected_score = dot_product_score * correction_factor
    return corrected_score

# Example usage
def example_secure_comparison_protocol():
    # Step 1: Generate example embeddings and convert to Q1.8 (simulate)
    embedding1 = np.random.uniform(-1, 1, size=128).astype(np.float32) * 50
    embedding2 = embedding1 + np.random.normal(0, 0.1, size=128)  # mated (genuine) sample

    X = float_to_q1_8(embedding1)
    Y = float_to_q1_8(embedding2)
    M = float_to_q1_8(np.random.uniform(-1, 1, size=128) * 50)  # random mask for X
    N = float_to_q1_8(np.random.uniform(-1, 1, size=128) * 50)  # random mask for Y

    # Step 2: Secure enrollment (encrypt templates)
    X_enc, M_enc, R1 = secure_enrollment(X, M)
    Y_enc, N_enc, R2 = secure_enrollment(Y, N)

    # Step 3: Distribute shares
    P1, P2 = distribute_template_shares(X_enc, Y_enc, M_enc, N_enc)

    # Step 4: Convert encrypted templates to binary and generate octets
    def int_to_bits(arr, bit_width=16):
        return np.unpackbits(arr.view(np.uint8)).reshape(-1, bit_width)[:, ::-1]

    # Use first 8 elements * 16 bits = 128 bits for demonstration
    octets_X = generate_octets(int_to_bits(P1['X'][:8]).flatten())
    octets_Y = generate_octets(int_to_bits(P2['Y'][:8]).flatten())
    octets_M = generate_octets(int_to_bits(P1['M'][:8]).flatten())
    octets_N = generate_octets(int_to_bits(P2['N'][:8]).flatten())

    # Step 5: Secure masked AND dot product
    raw_score = secure_and_masked_dot_product(P1, P2, octets_X, octets_Y, octets_M, octets_N)
    print("Raw secure masked dot product score:", raw_score)

    # Step 6: Compute observed Hamming distance between octets_X and octets_Y
    avg_bits_diff, observed_hd = compute_hamming_distance_bits(octets_X, octets_Y)
    print(f"Observed average bits difference per octet: {avg_bits_diff:.2f} bits (~{observed_hd*100:.1f}%)")

    # Step 7: Apply correction if observed HD deviates from expected 25%
    corrected_score = apply_correction(raw_score, observed_hd, expected_hd=0.25)
    print("Corrected similarity score:", corrected_score)

    # Step 8: Authentication decision
    threshold = 1000  # Example threshold, adjust based on system requirements
    if corrected_score >= threshold:
        print(f"Authentication ACCEPTED (score {corrected_score:.2f} >= threshold {threshold})")
    else:
        print(f"Authentication REJECTED (score {corrected_score:.2f} < threshold {threshold})")


if __name__ == "__main__":
    example_secure_comparison_protocol()
