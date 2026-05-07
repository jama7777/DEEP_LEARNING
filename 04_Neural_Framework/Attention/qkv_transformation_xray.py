import numpy as np

def qkv_transformation_xray():
    print("🎭 THE QKV TRANSFORMATION: ONE WORD, THREE PERSONALITIES")
    print("=" * 70)

    # 1. THE WORD (e.g., 'River' embedding)
    x = np.array([[1.0, 0.0, 0.5, -0.2]]) # 1x4 Vector
    
    print(f"Original Word Vector (River): {x}")

    # 2. THE THREE PERSONALITY MATRICES (Randomly initialized)
    # In a real model, these are learned during training
    Wq = np.random.randn(4, 4) # Query weights
    Wk = np.random.randn(4, 4) # Key weights
    Wv = np.random.randn(4, 4) # Value weights

    # 3. THE TRANSFORMATION (Dot Products)
    Q = np.dot(x, Wq) # The Question
    K = np.dot(x, Wk) # The ID Card
    V = np.dot(x, Wv) # The Knowledge

    print("\n" + "-" * 70)
    print(f"Query (Q) - What I'm looking for: \n{Q}")
    print(f"\nKey (K) - My Identity Card: \n{K}")
    print(f"\nValue (V) - My Secret Content: \n{V}")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("💡 THE DEEP TRUTH:")
    print("1. One word vector now has 3 different 'Views' of itself.")
    print("2. The Model learns the Q, K, V weights so it can decide which")
    print("   features of a word are important for 'Questions' vs 'Answers'.")
    print("3. This is the 'Secret Engine' that allows Transformers to handle")
    print("   complex grammar and context.")

if __name__ == "__main__":
    qkv_transformation_xray()
