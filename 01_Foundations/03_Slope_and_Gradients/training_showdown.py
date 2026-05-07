import numpy as np
from xray_utils import show_detailed_math, show_elementwise_logic

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def training_showdown():
    # Initial Conditions
    x, target, lr = 1.0, 1.0, 0.5
    w1, w2, w3 = 1.0, 1.0, 1.0
    
    print("--- TRAINING SHOWDOWN: X-RAY EDITION ---")
    
    for step in range(2): # Just show a few steps of math deep dive
        print(f"\n{'='*20} STEP {step} {'='*20}")
        
        # Calculate Predictions
        p1, p2, p3 = sigmoid(w1*x), sigmoid(w2*x), sigmoid(w3*x)
        
        # S1 Logic (Aggressive)
        gap1 = (p1 - target)
        grad1 = 2 * gap1 * p1 * x
        show_detailed_math("S1 GRADIENT (2 * Gap * Confidence)", [2 * gap1, p1, x], grad1)
        
        # S3 Logic (Balanced)
        gap3 = (p3 - target)
        sens3 = p3 * (1 - p3)
        grad3 = 2 * gap3 * sens3 * x
        show_detailed_math("S3 GRADIENT (2 * Gap * RealSensitivity)", [2 * gap3, sens3, x], grad3)
        
        # Update
        w1 -= lr * grad1; w3 -= lr * grad3

    print("\nCheck the terminal above! You can see exactly why S1 moves faster than S3.")

if __name__ == "__main__":
    training_showdown()
