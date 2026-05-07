import numpy as np

def full_attention_engine():
    print("🤝 THE FULL ATTENTION ENGINE: THE WORD DANCE")
    print("=" * 70)

    # 1. TWO WORDS: 'river' and 'bank'
    # Embeddings (dim=4)
    river = np.array([1.0, 0.0, 0.8, 0.0])
    bank  = np.array([0.9, 0.1, 0.7, 0.1])
    
    X = np.stack([river, bank]) # 2x4 Matrix
    
    # 2. Q, K, V WEIGHTS (Simulating learned weights)
    Wq = np.eye(4) * 0.5 # Simple identity-ish weights
    Wk = np.eye(4) * 0.5
    Wv = np.eye(4) * 1.0 # Value is the pure content
    
    # 3. GENERATE THE PERSONALITIES
    Q = np.dot(X, Wq) # 2x4
    K = np.dot(X, Wk) # 2x4
    V = np.dot(X, Wv) # 2x4

    # 4. THE SCORES (Q x K^T)
    # How much does every word like every other word?
    scores = np.dot(Q, K.T) # 2x2 matrix
    
    # 5. THE SCALE (Divide by sqrt of dimension)
    scaled_scores = scores / np.sqrt(4)
    
    # 6. THE VOTING (Softmax)
    # Turning scores into percentages
    exp_scores = np.exp(scaled_scores)
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 7. THE MIXING (Weights x V)
    # The final 'Context-Aware' embeddings
    output = np.dot(weights, V)

    print(f"Attention Weights (How much they talk):\n{weights}")
    print("\n--- 🏁 THE TRANSFORMATION ---")
    print(f"Original 'Bank' Vector: {bank}")
    print(f"Context-Aware 'Bank':   {output[1]}") # The second row is 'bank'
    
    print("\n" + "=" * 70)
    print("💡 THE DEEP TRUTH:")
    print("1. Look at the Context-Aware Bank! It has 'absorbed' some of the")
    print("   features of the word 'River'.")
    print("2. The model no longer sees 'Bank' as an isolated word.")
    print("3. It sees 'Bank-in-the-context-of-River'. This is intelligence.")

if __name__ == "__main__":
    full_attention_engine()
