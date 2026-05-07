# CHALLENGE: Fix the AI!
# Currently, the AI is "confused" and moves the weight in the WRONG direction.
# Find the mistake in the 'train_one_step' function.

def train_one_step(w, x, target, lr):
    prediction = w * x
    loss = (prediction - target) ** 2
    
    # --- FIXED ---
    # The derivative of (w*x - target)^2 with respect to w is:
    # 2 * (w*x - target) * x
    gradient = 2 * (prediction - target) * x 
    # -------------
    
    new_w = w - (lr * gradient)
    return new_w, loss, gradient, new_w

def main():
    w = 0.0
    x = 2.0
    target = 10.0
    lr = 0.2
    
    print("Starting training...")
    for i in range(30):
        w, loss, gradient, new_w = train_one_step(w, x, target, lr)
        
        # Print every 1 step since we reduced the range, but let's be smart
        print(f"Step {i+1}: Weight = {w:.6f} | Loss = {loss:.10f} | Gradient: {gradient:.10f} | new_w: {new_w:.10f} | w * x : {w * x:.10f}")
            

if __name__ == "__main__":
    main()
