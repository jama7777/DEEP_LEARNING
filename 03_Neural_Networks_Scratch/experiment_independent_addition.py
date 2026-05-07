import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_single_neuron(X, y, epochs=100000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    W = np.random.randn(X.shape[1], 1) # Automatically sizes to number of inputs
    b = np.zeros((1, 1))
    lr = 0.01
    
    for _ in range(epochs):
        z = np.dot(X, W) + b
        a = sigmoid(z)
        
        # Calculate gradients
        error = a - y
        dZ = error * a * (1 - a)
        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        
        # Update weights
        W -= lr * dW
        b -= lr * db
        
    return a # Return final predictions

def main():
    # The XOR Problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_true = np.array([[0], [1], [1], [0]])
    
    print("==================================================")
    print("🔬 EXPERIMENT: Adding Independently Trained Neurons")
    print("==================================================\n")
    
    print("1️⃣ Training Neuron 1 completely alone on XOR...")
    pred1 = train_single_neuron(X, y_true, seed=42)
    print("Neuron 1 gave up and guessed ~0.5 for everything:")
    print(np.round(pred1, 3))
    
    print("\n2️⃣ Training Neuron 2 completely alone on XOR (different random start)...")
    pred2 = train_single_neuron(X, y_true, seed=99)
    print("Neuron 2 also gave up and guessed ~0.5 for everything:")
    print(np.round(pred2, 3))
    
    print("\n3️⃣ Now, let's add their outputs together and pass through Sigmoid as you asked!")
    
    # 1. Add their numbers
    combined_addition = pred1 + pred2
    print("Added outputs (pred1 + pred2):")
    print(np.round(combined_addition, 3))
    
    # 2. Go with sigmoid after adding
    final_pred = sigmoid(combined_addition)
    
    print("\n🏁 FINAL RESULT (Sigmoid of the addition):")
    print(np.round(final_pred, 3))
    
    print("\nTargets were:")
    print(y_true)
    
    print("\n❌ STILL FAILED!")
    print("Adding two failed numbers (0.5 + 0.5 = 1.0) and passing through Sigmoid (0.731) does NOT solve XOR.")
    print("The magic is not the addition itself. The magic is BACKPROPAGATION coaching them to learn different things DURING training!")

if __name__ == "__main__":
    main()
