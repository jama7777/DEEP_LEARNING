import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def run_experiment(op_name, forward_fn, backward_fn, lr=0.1, epochs=5000):
    # Features: x1, x2, x1*x2, bias
    X = np.array([
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ])
    y_true = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    W1 = np.random.randn(4, 1) * 0.1
    
    for epoch in range(epochs):
        # --- NEW FORWARD MATH ---
        z1 = forward_fn(X, W1)
        predictions = sigmoid(z1)
        
        # --- NEW BACKWARD CALCULUS ---
        gap = predictions - y_true
        d_z1 = 2 * gap * sigmoid_derivative(z1)
        
        derivative = backward_fn(X, W1)
        d_W1_matrix = d_z1 * derivative
        d_W1 = np.sum(d_W1_matrix, axis=0, keepdims=True).T
        
        W1 -= lr * d_W1
        
    return predictions

def main():
    print("--- 👽 THE ALIEN NEURON EXPERIMENT: 5 ALTERNATE UNIVERSES ---")
    print("Targets for XOR are: 0, 1, 1, 0\n")
    
    # 1. ADDITION (z = x + w)
    def add_fwd(X, W): return np.sum(X + W.T, axis=1, keepdims=True)
    def add_bwd(X, W): return np.ones_like(X) # Calculus: Derivative of (x+w) is 1
    
    # 2. SUBTRACTION (z = x - w)
    def sub_fwd(X, W): return np.sum(X - W.T, axis=1, keepdims=True)
    def sub_bwd(X, W): return -np.ones_like(X) # Calculus: Derivative of (x-w) is -1
    
    # 3. ABSOLUTE DISTANCE (z = |x - w|)
    def abs_fwd(X, W): return np.sum(np.abs(X - W.T), axis=1, keepdims=True)
    def abs_bwd(X, W): return -np.sign(X - W.T) # Calculus: Derivative is -1 or 1
    
    # 4. SINE WAVE (z = sin(x * w))
    def sin_fwd(X, W): return np.sum(np.sin(X * W.T), axis=1, keepdims=True)
    def sin_bwd(X, W): return X * np.cos(X * W.T) # Calculus: Chain Rule!
    
    # 5. EXPONENTIAL (z = x * e^w) -> Forces weights to be strictly positive
    def exp_fwd(X, W): 
        W_safe = np.clip(W.T, -50, 50)
        return np.sum(X * np.exp(W_safe), axis=1, keepdims=True)
    def exp_bwd(X, W): 
        W_safe = np.clip(W.T, -50, 50)
        return X * np.exp(W_safe)

    experiments = [
        ("1. ADDITION   (z = x + w)    ", add_fwd, add_bwd, 0.05),
        ("2. SUBTRACTION(z = x - w)    ", sub_fwd, sub_bwd, 0.05),
        ("3. ABS DIST.  (z = |x - w|)  ", abs_fwd, abs_bwd, 0.05),
        ("4. SINE WAVE  (z = sin(x*w)) ", sin_fwd, sin_bwd, 0.5),
        ("5. EXPONENTIAL(z = x * e^w)  ", exp_fwd, exp_bwd, 0.01)
    ]

    for name, fwd, bwd, lr in experiments:
        try:
            preds = run_experiment(name, fwd, bwd, lr)
            print(f"{name} -> [0,0]: {preds[0][0]:.2f} | [0,1]: {preds[1][0]:.2f} | [1,0]: {preds[2][0]:.2f} | [1,1]: {preds[3][0]:.2f}")
        except Exception as e:
            print(f"{name} -> FAILED: {e}")

if __name__ == "__main__":
    main()
