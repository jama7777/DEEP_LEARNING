import numpy as np

def compare_losses_proportional(w, b, x):
    """ What you called 'Normal Loss' - actually a linear gradient """
    y_true = 10
    l_r = 0.001
    for i in range(20):
        prediction = x * w + b
        error = prediction - y_true
        w -= l_r * (error * x)
        b -= l_r * error
        print(f"Epoch {i}: Pred={prediction:.4f}, Error={error:.4f}")
    print("#" * 50)
    return w, b, x

def compare_losses_square(w, b, x):
    """ Squared Loss - naturally slows down """
    y_true = 10
    l_r = 0.001
    for i in range(20):
        prediction = x * w + b
        error = prediction - y_true
        w -= l_r * (2 * error * x)
        b -= l_r * error

        print(f"Epoch {i}: Pred={prediction:.4f}, Error={error:.4f}")
    return w, b, x

def compare_losses_absolute(w, b, x):
    """ TRUE Absolute Loss - Constant Speed (The Drawback!) """
    y_true = 10
    l_r = 0.3 # Non-dividing LR to show vibration
    print(f"\n--- Starting TRUE Absolute Loss (Constant Speed) for x={x} ---")
    for i in range(20):
        prediction = x * w + b
        error = prediction - y_true
        
        # DRAWBACK: Gradient is ONLY the direction (+1 or -1).
        # It doesn't know how close it is!
        gradient = np.sign(error) * x
        
        w -= l_r * gradient
        b -= l_r * np.sign(error)
        
        print(f"Epoch {i}: Pred={prediction:.4f}, Error={error:.4f}, gradient={gradient}")
    return w, b, x

if __name__ == "__main__":
    x_val = 9
    w1, b1, _ = compare_losses_proportional(0.0, 0.0, x_val)
    w2, b2, _ = compare_losses_square(0.0, 0.0, x_val)
    w3, b3, _ = compare_losses_absolute(0.0, 0.0, x_val)
    
    print("\n" + "="*40)
    print(f"1. Proportional Pred: {x_val * w1 + b1:.8f} (Settled)")
    print(f"2. Squared Pred:      {x_val * w2 + b2:.8f} (Settled)")
    print(f"3. Absolute Pred:     {x_val * w3 + b3:.8f} (Vibrating!)")
