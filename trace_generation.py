import numpy as np

#############
# binary de kaç tane 1 var 
def hw(a, dtype='uint32', B=None):  
    b = np.zeros(a.shape, dtype=dtype)  # a boyutunda 0 lar dizisi 
    if B is None:   
        B = np.dtype(dtype).itemsize*8
    for i in range(B):  # B deki bitlere bakarak b ye 1 ekler 
        b += (a.astype(dtype) >> i) & 0x1
    return b
############

# verilen mesajdan sample lar oluşturuyor random katsayılar oluşturmak amaç 
# mod q ;  sigma noise ; k katsayı sayısı
#mesaj formati list of bytes 
def compute_coeffs(msg, q=3329, sigma=0, k=0):

    # Initialize the output list
    coeffs = [0] * (32 * 8) # 256 tane 0 liste 32 tane 1 byte
    # random sample oluşturuyor 
    samples = np.random.randint(size=256*(k+1), low=0, dtype='int32', high=2**31)
    
    for i in range(32):  #iteare byte messages 
        for j in range(8):  # iterate bit 

            # Extract the j-th bit of msg[i]
            bit = (msg[i] >> j) & 1

            # mask will be either 0 (if bit == 0) or -1 (if bit == 1)
            mask = -bit
            samples[(8*i + j)*(k+1)] = mask
            
            # Use integer division if q is an integer
            # 
            coeffs[8 * i + j] = mask & ((q + 1) // 2)
            if k > 0:
                samples[(8*i + j)*(k+1)+1] = coeffs[8 * i + j]

    samples = hw(samples) + np.random.normal(0, sigma, 256*(k+1))
    return samples

############
# Step 1: Reference Phase - Profile the reference traces on a per-bit basis.
def profile_reference_trace(r1, r0):
    """
    For each bit (total 256), find the index of the maximum and minimum in the 
    corresponding block of samples (the block length is len(r1)//256). Then compute 
    a global threshold T as the average of the differences between r0 and r1 at these indices.
    """
    block_length = len(r1) // 256  # each bit has a block of samples of this length
    PoImax = []
    PoImin = []
    for i in range(256):
        block_r1 = r1[i*block_length:(i+1)*block_length]
        max_idx = np.argmax(block_r1) + i*block_length
        min_idx = np.argmin(block_r1) + i*block_length
        PoImax.append(max_idx)
        PoImin.append(min_idx)
    
    # Compute the difference at each PoI for both reference traces
    a0 = np.array([r0[PoImax[i]] - r0[PoImin[i]] for i in range(256)])
    a1 = np.array([r1[PoImax[i]] - r1[PoImin[i]] for i in range(256)])
    T = 0.5 * (np.mean(a0) + np.mean(a1))
    
    return PoImin, PoImax, T

############
# Step 2: Attack Phase - Recover Message from the Target Trace (bit-by-bit).
def recover_message(PoImin, PoImax, T, p):
    """
    Recover the message bit-by-bit. For each bit, compare the difference between 
    the PoI max and min in the target trace with threshold T.
    """
    recovered_bits = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        delta = p[PoImax[i]] - p[PoImin[i]]
        recovered_bits[i] = 1 if delta > T else 0
    return recovered_bits

############
# Helper function: Convert a 32-byte message to a 256-bit array.
# (Extract bits in the same order as in compute_coeffs: LSB first in each byte.)
def message_to_bits(message):
    bits = []
    for byte in message:
        for j in range(8):
            bits.append((byte >> j) & 1)
    return np.array(bits, dtype=np.uint8)

############
# Example usage:

# Create a random 32-byte message (values between 0 and 255)
message = np.random.randint(low=0, high=256, size=32, dtype='int32')

# Generate reference traces:
# Here, samples1 corresponds to a message with all 1s (i.e. [255]*32),
# and samples0 corresponds to a message with all 0s ([0]*32).
samples1 = compute_coeffs([255]*32, sigma=4, k=9)
samples0 = compute_coeffs([0]*32, sigma=4, k=9) 
# Generate the target trace for the actual message
samples_r = compute_coeffs(message, sigma=4, k=9)

# Profile the reference traces (now per-bit)
PoImin, PoImax, T = profile_reference_trace(samples1, samples0)

# Recover the message bit-by-bit from the target trace
recovered_message_bits = recover_message(PoImin, PoImax, T, samples_r)

# Convert the original message to bits (using the same LSB-first order)
original_message_bits = message_to_bits(message)

# Compare original and recovered bits
print("Original message bits:")
print(original_message_bits)
print("Recovered message bits:")
print(recovered_message_bits)
print("Number of bit errors:", np.sum(original_message_bits != recovered_message_bits))


# countermeasure olduğunda nasıl çalışıyor
# find message space