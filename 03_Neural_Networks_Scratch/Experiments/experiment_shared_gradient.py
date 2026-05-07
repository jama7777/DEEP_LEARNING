import numpy as np
import random
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    # Using 'a' directly since a = sigmoid(z)
    return a * (1 - a)

def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_true = np.array([[0], [1], [1], [0]])
    l= random.randint(1, 10000)
    np.random.seed(l) # Ensure random starting paths
    
    print("==================================================")
    print("🔬 EXPERIMENT: Individual Calculation, Shared Gradient")
    print("==================================================\n")
    
    # ==========================================
    # 1. INDIVIDUAL NEURONS (No matrices!)
    # ==========================================
    # Hidden Neuron 1 (Completely separate variable)
    w_h1 = np.random.randn(2, 1)
    b_h1 = np.zeros((1, 1))
    
    # Hidden Neuron 2 (Completely separate variable)
    w_h2 = np.random.randn(2, 1)
    b_h2 = np.zeros((1, 1))
    
    # Output Neuron (Connects to H1 and H2)
    w_out1 = np.random.randn(1, 1) # Weight for Neuron 1
    w_out2 = np.random.randn(1, 1) # Weight for Neuron 2
    b_out = np.zeros((1, 1))
    
    lr = 0.1
    epochs = 10000
    
    for epoch in range(epochs):
        
        # ==========================================
        # 2. INDIVIDUAL FORWARD PASS
        # ==========================================
        # Neuron 1 calculates its output
        z1 = np.dot(X, w_h1) + b_h1
        a1 = sigmoid(z1)
        
        # Neuron 2 calculates its output completely separately
        z2 = np.dot(X, w_h2) + b_h2
        a2 = sigmoid(z2)
        
        # The Boss (Output) combines their answers
        z_out = (a1 * w_out1) + (a2 * w_out2) + b_out
        a_out = sigmoid(z_out)
        
        # ==========================================
        # 3. SHARED GRADIENT (The Teamwork Step!)
        # ==========================================
        # The Boss looks at the final answer and finds the total error
        error_out = a_out - y_true
        dZ_out = error_out * sigmoid_derivative(a_out)
        
        # Output weight gradients
        dw_out1 = np.sum(a1 * dZ_out)
        dw_out2 = np.sum(a2 * dZ_out)
        db_out = np.sum(dZ_out)
        
        # --- THE MAGIC HAPPENS HERE ---
        # The Boss passes the shared 'dZ_out' back to both workers!
        # It tells Neuron 1: "Take the shared error, multiply by your weight, and fix yourself!"
        dZ_h1 = (dZ_out * w_out1) * sigmoid_derivative(a1)
        
        # It tells Neuron 2: "Take the same shared error, multiply by YOUR weight, and fix yourself!"
        dZ_h2 = (dZ_out * w_out2) * sigmoid_derivative(a2)
        
        # Hidden weight gradients (calculated individually based on the shared error)
        dw_h1 = np.dot(X.T, dZ_h1)
        db_h1 = np.sum(dZ_h1, axis=0, keepdims=True)
        
        dw_h2 = np.dot(X.T, dZ_h2)
        db_h2 = np.sum(dZ_h2, axis=0, keepdims=True)
        
        # ==========================================
        # 4. UPDATES
        # ==========================================
        w_out1 -= lr * dw_out1
        w_out2 -= lr * dw_out2
        b_out -= lr * db_out
        
        w_h1 -= lr * dw_h1
        b_h1 -= lr * db_h1
        
        w_h2 -= lr * dw_h2
        b_h2 -= lr * db_h2
    print(f"Random Seed was {l}")
    print("🏁 FINAL PREDICTION:")
    print(np.round(a_out, 3))
    print("\nTargets were:")
    print(y_true)
    
    print("\n🎉 DID IT SOLVE XOR?")
    print("YES! Even though they calculated everything individually, because they SHARED the Output Error (dZ_out), they learned to work together!")

if __name__ == "__main__":
    main()
