import numpy as np 

# 1. THE RAW DATA
# Imagine this is a word embedding or hidden state
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
print(f"1. Raw Data: {x}")

# 2. THE MEAN (Finding the 'Balance Point')
mean = np.mean(x)
print(f"2. Mean (Average): {mean}")

# 3. CENTERING (Subtracting the Mean)
# This forces the numbers to live around Zero.
centered = x - mean
print(f"3. Centered (at 0): {centered}")

# 4. THE VARIANCE (Finding the 'Spread')
# How far, on average, are the numbers from the mean?
var = np.var(x)
std = np.sqrt(var)
print(f"4. Spread (Std Dev): {std:.2f}")

# 5. SCALING (Dividing by Std Dev)
# This forces the spread to be exactly 1.0.
normalized = centered / std
print(f"\n--- 🏁 THE FINAL EQUALIZED DATA ---")
print(normalized)
print(f"New Mean: {np.mean(normalized):.1f}")
print(f"New Std:  {np.std(normalized):.1f}")

print("\n💡 WHY THIS HELPS (THE DEEP TRUTH):")
print("1. STABILITY: No matter if your inputs are [1, 2, 3] or [1000, 2000, 3000],")
print("   the output of LayerNorm will ALWAYS be roughly [-1.2, 0, 1.2].")
print("2. SPEED: The next layer doesn't have to 'search' for the right weights")
print("   for massive numbers. It can focus on the patterns in the small, stable numbers.")
print("3. GRADIENT HEALTH: Just like ReLU prevents dying gradients, Normalization")
print("   prevents EXPLODING gradients by keeping the activations small.")
