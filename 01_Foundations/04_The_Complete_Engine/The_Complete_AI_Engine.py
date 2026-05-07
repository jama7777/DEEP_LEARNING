import numpy as np
import time

# ==========================================
# THE COMPLETE AI ENGINE (From Foundations)
# ==========================================

def sigmoid(z):
    """The Gatekeeper: Squashes any number to 0 or 1"""
    return 1 / (1 + np.exp(-z))

def train_step(x, weight, bias, target, learning_rate):
    """A single cycle of AI thinking and learning"""
    
    # 1. FORWARD PASS (The "Guess")
    # Score = (Input * Weight) + Bias
    score = (x * weight) + bias
    # Prediction = Squashed Score
    prediction = sigmoid(score)
    
    # 2. CALCULATE LOSS (The "Discomfort")
    # How far is our 0.0-1.0 guess from the actual 0.0 or 1.0 answer?
    loss = (prediction - target) ** 2
    
    # 3. CALCULATE GRADIENT (The "Compass")
    # We need to know which way to turn the knobs.
    # This math tells us the slope of the error.
    error_signal = 2 * (prediction - target) * (prediction * (1 - prediction))
    
    gradient_w = error_signal * x
    gradient_b = error_signal * 1
    
    # 4. WEIGHT UPDATE (The "Nudge")
    # Move the knobs slightly in the right direction
    new_w = weight - (learning_rate * gradient_w)
    new_b = bias - (learning_rate * gradient_b)
    
    return new_w, new_b, loss, prediction

def main():
    print("--- Training a 1-Neuron AI ---")
    print("Goal: Learn that if Input=1.0, the Answer is YES (1.0)\n")
    
    # Starting with random 'wrong' knobs
    w = 0.0
    b = 0.0
    x = 1.0
    target = 1.0
    lr = 1.0 # High learning rate to see it fast
    
    print(f"{'Epoch':<6} | {'Weight':<10} | {'Bias':<10} | {'Prediction':<12} | {'Loss':<10}")
    print("-" * 65)
    
    for epoch in range(1, 21):
        w, b, loss, pred = train_step(x, w, b, target, lr)
        
        # Only print every 2nd step to keep it clean
        if epoch % 2 == 0 or epoch == 1:
            print(f"{epoch:<6} | {w:<10.2f} | {b:<10.2f} | {pred:<12.2f} | {loss:<10.4f}")
        
        time.sleep(0.1)

    print("\nResult:")
    print(f"Final Prediction: {pred:.4f} (Very close to 1.0!)")
    print("The AI successfully adjusted its 'knobs' to give the right answer.")

if __name__ == "__main__":
    main()
