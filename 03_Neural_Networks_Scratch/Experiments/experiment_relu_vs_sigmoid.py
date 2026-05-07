import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def leaky_relu(z):
    return np.where(z > 0, z, 0.01 * z)

def leaky_relu_derivative(z):
    return np.where(z > 0, 1.0, 0.01)

def run_deep_experiment(activation_type, X, y, epochs=10000, lr=0.1):
    # Architecture: 2 -> 4 -> 4 -> 4 -> 1
    np.random.seed(42)
    W1 = np.random.randn(2, 4) * np.sqrt(2/2); b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 4) * np.sqrt(2/4); b2 = np.zeros((1, 4))
    W3 = np.random.randn(4, 4) * np.sqrt(2/4); b3 = np.zeros((1, 4))
    W4 = np.random.randn(4, 1) * np.sqrt(2/4); b4 = np.zeros((1, 1))

    if activation_type == "sigmoid":
        act, act_deriv = sigmoid, sigmoid_derivative
    elif activation_type == "relu":
        act, act_deriv = relu, relu_derivative
    else:
        act, act_deriv = leaky_relu, leaky_relu_derivative

    first_layer_grads = []

    for epoch in range(epochs):
        # Forward
        a1 = act(np.dot(X, W1) + b1)
        a2 = act(np.dot(a1, W2) + b2)
        a3 = act(np.dot(a2, W3) + b3)
        z4 = np.dot(a3, W4) + b4
        predictions = sigmoid(z4)

        # Backward
        gap = predictions - y
        d_z4 = 2 * gap * sigmoid_derivative(z4)
        
        d_a3 = np.dot(d_z4, W4.T)
        d_z3 = d_a3 * act_deriv(np.dot(a2, W3) + b3)
        
        d_a2 = np.dot(d_z3, W3.T)
        d_z2 = d_a2 * act_deriv(np.dot(a1, W2) + b2)
        
        d_a1 = np.dot(d_z2, W2.T)
        d_z1 = d_a1 * act_deriv(np.dot(X, W1) + b1)

        first_layer_grads.append(np.mean(np.abs(d_z1)))

        # Update
        W4 -= lr * np.dot(a3.T, d_z4); b4 -= lr * np.sum(d_z4, axis=0)
        W3 -= lr * np.dot(a2.T, d_z3); b3 -= lr * np.sum(d_z3, axis=0)
        W2 -= lr * np.dot(a1.T, d_z2); b2 -= lr * np.sum(d_z2, axis=0)
        W1 -= lr * np.dot(X.T, d_z1); b1 -= lr * np.sum(d_z1, axis=0)

    return predictions, first_layer_grads

def main():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    print("--- ⚔️ THE GREAT BATTLE: SIGMOID vs RELU vs LEAKY RELU ---")
    sig_preds, sig_grads = run_deep_experiment("sigmoid", X, y)
    relu_preds, relu_grads = run_deep_experiment("relu", X, y)
    leaky_preds, leaky_grads = run_deep_experiment("leaky", X, y)

    print(f"{'Activation':<15} | {'[0,0]':<6} | {'[0,1]':<6} | {'[1,0]':<6} | {'[1,1]':<6} | {'Gradient Health'}")
    print("-" * 90)
    def fmt(p): return f"{p[0]:.3f}"
    print(f"{'SIGMOID':<15} | {fmt(sig_preds[0]):<6} | {fmt(sig_preds[1]):<6} | {fmt(sig_preds[2]):<6} | {fmt(sig_preds[3]):<6} | {sig_grads[-1]:.8f}")
    print(f"{'RELU':<15} | {fmt(relu_preds[0]):<6} | {fmt(relu_preds[1]):<6} | {fmt(relu_preds[2]):<6} | {fmt(relu_preds[3]):<6} | {relu_grads[-1]:.8f}")
    print(f"{'LEAKY RELU':<15} | {fmt(leaky_preds[0]):<6} | {fmt(leaky_preds[1]):<6} | {fmt(leaky_preds[2]):<6} | {fmt(leaky_preds[3]):<6} | {leaky_grads[-1]:.8f}")

if __name__ == "__main__":
    main()
