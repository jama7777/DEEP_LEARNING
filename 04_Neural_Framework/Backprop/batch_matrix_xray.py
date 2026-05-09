import numpy as np

def batch_matrix_xray():
    print("🥞 THE BATCH MATRIX X-RAY: PROCESSING 'THE' AND 'SUN' TOGETHER")
    print("=" * 75)

    # 1. THE BATCH INPUT (2 rows of 4D embeddings)
    # Row 0: 'the'
    # Row 1: 'sun'
    X = np.array([
        [0.1, 0.2, -0.1, 0.5], # 'the'
        [0.8, -0.1, 0.2, 0.1]  # 'sun'
    ])
    print(f"INPUT MATRIX (X) - 2 Words Stacked:\n{X}")

    # 2. THE SHARED WEIGHTS (4 inputs -> 3 vocab scores)
    # These same numbers will be used for BOTH words
    W = np.array([
        [0.5, 0.1, 0.2],
        [0.2, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.4, 0.1, 0.3]
    ])
    print(f"\nSHARED WEIGHTS (W):\n{W}")

    # 3. THE BATCH FORWARD (Matrix Multiplication)
    # The computer does both words in ONE shot
    Y = np.dot(X, W)
    
    print("\n" + "-" * 75)
    print("FORWARD RESULT (Y = X * W):")
    print(f"Prediction for 'the' (Row 0): {Y[0]}")
    print(f"Prediction for 'sun' (Row 1): {Y[1]}")

    # 4. THE BATCH TARGETS
    # Pair 1: 'the' -> 'sun' (Target Index 1)
    # Pair 2: 'sun' -> 'rises' (Target Index 2)
    T = np.array([
        [0, 1, 0], # Target for 'the' is index 1
        [0, 0, 1]  # Target for 'sun' is index 2
    ])

    # 5. THE BATCH ERROR (P - Y)
    # Again, we do the subtraction for the WHOLE PANCAKE at once
    # (Assuming Softmax was already done)
    error = Y - T 
    print(f"\nBATCH ERROR MATRIX (Error = Y - T):\n{error}")

    print("\n" + "-" * 75)
    print("💡 THE DEEP TRUTH:")
    print("1. We used the SAME weights (W) for both rows.")
    print("2. The computer doesn't 'know' there are two words; it just sees a matrix.")
    print("3. Because they are in the same matrix, we can average their errors")
    print("   instantly to update the weights for the next round.")

if __name__ == "__main__":
    batch_matrix_xray()
