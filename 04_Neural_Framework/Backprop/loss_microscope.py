import numpy as np

def cross_entropy_microscope():
    print("🔬 THE LOSS MICROSCOPE: MEASURING REGRET (Cross-Entropy)")
    print("=" * 65)

    # 1. THE LOGITS (Raw Output Scores)
    # The brain 'shouts' these numbers
    logits = np.array([2.0, 1.0, 0.1]) # Scores for [ai, love, deep]
    target_idx = 2 # The real answer was 'deep' (Index 2)
    
    print(f"Logits (Raw Brain Output): {logits}")
    print(f"Target Word Index: {target_idx} ('deep')")

    # 2. SOFTMAX (The Probability Filter)
    exp_logits = np.exp(logits - np.max(logits)) # Stability trick
    probs = exp_logits / np.sum(exp_logits)
    
    print("\n--- 🔦 THE SOFTMAX (Percentages) ---")
    print(f"Probabilities: {probs}")
    print(f"The model is {probs[0]*100:.1f}% sure it is 'ai'.")
    print(f"The model is {probs[target_idx]*100:.1f}% sure it is 'deep'.")

    # 3. THE LOSS (The Regret)
    # Formula: -log(probability of correct class)
    correct_prob = probs[target_idx]
    loss = -np.log(correct_prob)

    print("\n--- 🎯 THE CROSS-ENTROPY LOSS (The Pain) ---")
    print(f"Correct Probability: {correct_prob:.4f}")
    print(f"Loss = -log({correct_prob:.4f}) = {loss:.4f}")

    # 4. THE GRADIENT (The Complaint)
    # The beauty of Cross-Entropy: Gradient = Probs - Target
    target_one_hot = np.zeros_like(probs)
    target_one_hot[target_idx] = 1.0
    
    gradient = probs - target_one_hot
    
    print("\n--- 🕵️ THE GRADIENT (The Blame) ---")
    print(f"Probs:  {probs}")
    print(f"Target: {target_one_hot}")
    print(f"Blame:  {gradient}")
    
    print("\n" + "=" * 65)
    print("💡 THE VERBAL SUMMARY:")
    print("1. The model was 'Overconfident' about index 0 (ai).")
    print("2. The Gradient for index 0 is positive (+0.6), saying 'SHUT UP!'")
    print("3. The Gradient for index 2 is negative (-0.9), saying 'SPEAK LOUDER!'")
    print("4. This is much better than MSE because it focuses only on the probabilities.")

if __name__ == "__main__":
    cross_entropy_microscope()
