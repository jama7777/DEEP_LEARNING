import numpy as np
from xray_utils import show_detailed_math, show_activation_logic, show_elementwise_logic

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def investigate_gradient(x, w, b, target):
    # --- THE FORWARD PASS ---
    score = (x * w) + b
    pred = sigmoid(score)
    loss = (pred - target) ** 2
    
    # --- X-RAY VISION ---
    show_detailed_math("1. SCORE (x * w + b)", [x, w, b], score, operation="+")
    show_activation_logic("PREDICTION (sigmoid(score))", score, pred, "sigmoid")
    
    # --- THE GRADIENT DETECTIVE ---
    gap = (pred - target)
    show_elementwise_logic("GAP (Pred - Target)", np.array([[pred]]), np.array([[target]]), np.array([[gap]]), "-")
    
    sensitivity = pred * (1 - pred)
    show_activation_logic("SENSITIVITY (p * (1-p))", pred, sensitivity, "deriv_sigmoid")
    
    error_signal = 2 * gap * sensitivity
    grad_w = error_signal * x
    grad_b = error_signal * 1
    
    show_detailed_math("2. GRAD_W (ErrorSignal * x)", [error_signal, x], grad_w)
    
    return pred, loss, grad_w, grad_b, gap, sensitivity

def main():
    print("--- THE GRADIENT DETECTIVE: X-RAY EDITION ---")
    
    w, b = 1.0, 0.0
    x, target = 1.0, 1.0
    
    # We show just a few steps to see the math clearly
    for i in range(2):
        print(f"\n{'='*20} STEP {i} {'='*20}")
        pred, loss, grad_w, grad_b, gap, sens = investigate_gradient(x, w, b, target)
        
        # Take a step
        w -= 1.0 * grad_w
        b -= 1.0 * grad_b

    print("\nObservation:")
    print("Now you can see exactly how the 'Blame' flows from the Target, through the Sigmoid, back to the Weight!")

if __name__ == "__main__":
    main()
