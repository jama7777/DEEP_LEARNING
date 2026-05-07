import time

def train_one_step(w, x, target, learning_rate):
    # 1. Forward Pass (Prediction)
    prediction = w * x
    
    # 2. Calculate Loss (MSE)
    loss = (prediction - target) ** 2
    
    # 3. Calculate Gradient (The Slope)
    # The derivative of (w*x - target)^2 with respect to w is:
    # 2 * (prediction - target) * x
    gradient = 2 * (prediction - target) * x
    
    # 4. Update the Weight (The Nudge)
    # We move in the OPPOSITE direction of the gradient
    new_w = w - (learning_rate * gradient)
    
    return new_w, loss, gradient

def main():
    print("--- Gradient Descent Simulation ---")
    print("Goal: Find the weight 'w' so that (w * 2) = 10")
    print("Correct Answer is obviously 5.0\n")
    
    w = 0.0      # Start with a blind guess
    x = 2.0      # Input
    target = 10.0 # Target Answer
    lr = 0.1     # Learning Rate (Step Size)
    
    print(f"{'Step':<5} | {'Weight':<10} | {'Prediction':<12} | {'Loss':<12} | {'Action'}")
    print("-" * 65)
    
    for i in range(15):
        old_w = w
        w, loss, gradient = train_one_step(w, x, target, lr)
        
        # Decide action string for visualization
        action = "Increasing W" if w > old_w else "Decreasing W"
        if abs(loss) < 0.01: action = "Goal Reached! 🎉"
        
        print(f"{i+1:<5} | {old_w:<10.2f} | {old_w*x:<12.2f} | {loss:<12.2f} | {action}")
        
    print(f"\nFinal Weight: {w:.4f}")
    print("Notice how the steps get smaller as we get closer to the bottom of the valley!")

if __name__ == "__main__":
    main()
