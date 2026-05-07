import numpy as np
from xray_utils import show_detailed_math, show_dot_logic, show_activation_logic, show_elementwise_logic

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def main():
    # 1. THE DATA
    X = np.array([[0.9, 0.8, 0.2], [0.7, 0.7, 0.3], [0.3, 0.3, 0.9]])
    y_true = np.array([[1.0], [1.0], [0.0]])

    np.random.seed(42)
    W_h = np.random.randn(3, 4); b_h = np.zeros((1, 4))
    W_o = np.random.randn(4, 1); b_o = np.zeros((1, 1))
    
    learning_rate = 0.5
    
    print("--- FULL TRAINING: X-RAY EDITION ---")
    
    # We'll just show the deep dive for Epoch 0
    for epoch in range(1):
        # 1. Forward Propagation
        h_scores = np.dot(X, W_h) + b_h
        show_detailed_math("1. HIDDEN SCORES (X * W_h + b)", [X, W_h, b_h], h_scores)
        
        h_out = sigmoid(h_scores)
        show_activation_logic("HIDDEN ACTIVATIONS", h_scores, h_out, "sigmoid")
        
        f_scores = np.dot(h_out, W_o) + b_o
        show_detailed_math("2. FINAL SCORES (A_h * W_o + b)", [h_out, W_o, b_o], f_scores)
        
        preds = sigmoid(f_scores)
        show_activation_logic("PREDICTIONS", f_scores, preds, "sigmoid")
        
        # 2. Backpropagation
        print("\n" + "!"*20 + " BACKPROPAGATION " + "!"*20)
        
        err_o = preds - y_true
        show_elementwise_logic("OUTPUT ERROR (Pred - Target)", preds, y_true, err_o, "-")
        
        d_o = err_o * sigmoid_derivative(f_scores)
        show_detailed_math("3. d_out (Error * Sig_Deriv)", [err_o, sigmoid_derivative(f_scores)], d_o)
        
        err_h = np.dot(d_o, W_o.T)
        show_detailed_math("4. HIDDEN ERROR (d_o * W_o.T)", [d_o, W_o.T], err_h)
        
        d_h = err_h * sigmoid_derivative(h_scores)
        show_elementwise_logic("d_hidden (Error * Sens)", err_h, sigmoid_derivative(h_scores), d_h, "*")
        
        # 3. Update
        grad_Wo = np.dot(h_out.T, d_o)
        show_detailed_math("5. GRAD_W_output (A_h.T * d_o)", [h_out.T, d_o], grad_Wo)
        
        W_o -= learning_rate * grad_Wo
        W_h -= learning_rate * np.dot(X.T, d_h)

    print("\nTraining logic verified. You can now see every detail in the matrices!")

if __name__ == "__main__":
    main()
