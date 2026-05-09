import numpy as np

class DecisionChamber:
    def forward(self, logits):
        # 1. SOFTMAX: The Exponential Pressure Cooker
        # We subtract max for numerical stability (prevents overflow)
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        self.probs = probs
        return probs

    def calculate_loss(self, probs, target_idx):
        # 2. CROSS-ENTROPY: The Surprise Meter
        # Loss = -log(Probability of the correct word)
        loss = -np.log(probs[target_idx] + 1e-15)
        return loss

    def backward(self, target_idx):
        # 3. THE MIRACLE GRADIENT
        # dL/dLogits = Probs - Truth
        # This is the 'Baton' we pass back to the rest of the model
        d_logits = self.probs.copy()
        d_logits[target_idx] -= 1.0 # Subtract 1.0 from the correct word
        return d_logits

def softmax_ce_deep_xray():
    print("🔥 THE DECISION CHAMBER: SOFTMAX + CROSS-ENTROPY X-RAY")
    print("=" * 65)

    # 1. SETUP
    vocab = ["the", "sun", "moon", "rises"]
    logits = np.array([2.0, 5.0, 1.0, -1.0]) # Raw scores from the StableBlock
    target_word = "sun"
    target_idx = vocab.index(target_word)
    
    chamber = DecisionChamber()

    # 2. FORWARD PASS (Turning Scores into Probabilities)
    probs = chamber.forward(logits)
    print("STEP 1: RAW SCORES -> PROBABILITIES")
    for i, w in enumerate(vocab):
        print(f"   {w:5}: {logits[i]:>5.1f}  --->  {probs[i]*100:>5.1f}%")

    # 3. CALCULATE LOSS
    loss = chamber.calculate_loss(probs, target_idx)
    print(f"\nSTEP 2: CROSS-ENTROPY LOSS (The Surprise): {loss:.4f}")

    # 4. BACKWARD PASS (The Miracle Subtraction)
    d_logits = chamber.backward(target_idx)
    print("\nSTEP 3: THE MIRACLE BACKPROP (Probs - Truth)")
    print(f"Incoming Error to the StableBlock:\n{d_logits}")

    print("\n" + "-" * 65)
    print("💡 THE DEEP TRUTH:")
    print(f"1. For 'sun' (Correct): Probs ({probs[target_idx]:.2f}) - Truth (1.0) = {d_logits[target_idx]:.2f}")
    print(f"2. For 'the' (Wrong):   Probs ({probs[0]:.2f}) - Truth (0.0) = {d_logits[0]:.2f}")
    print("\nRESULT: The error signal is just the 'Gap' between reality and belief.")
    print("This is the most elegant baton-pass in all of AI!")

if __name__ == "__main__":
    softmax_ce_deep_xray()
