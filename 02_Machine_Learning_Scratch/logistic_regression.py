import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def generate_classification_data():
    """Create data where temp > 37.5 is a fever (1), otherwise (0)."""
    np.random.seed(42)
    # 100 people with temps between 35 and 40
    X = 35 + 5 * np.random.rand(100, 1)
    # Secret rule: Fever if temp > 37.5
    y = (X > 37.5).astype(float)
    return X, y

def train_logistic_regression(X, y, lr=0.1, epochs=2000):
    m = len(X)
    # Normalize X for better training
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_norm = (X - X_mean) / X_std
    
    # Initialize Weight and Bias
    w = np.random.randn(1, 1)
    b = np.zeros((1, 1))
    
    for epoch in range(epochs):
        # 1. Forward Pass
        score = X_norm.dot(w) + b
        prediction = sigmoid(score)
        
        # 2. Calculate Binary Cross-Entropy Loss
        # We add a tiny 1e-15 to avoid log(0) errors
        loss = -np.mean(y * np.log(prediction + 1e-15) + (1 - y) * np.log(1 - prediction + 1e-15))
        
        # 3. Calculate Gradients (Simplified for Cross-Entropy!)
        error = prediction - y
        gradient_w = (1/m) * X_norm.T.dot(error)
        gradient_b = (1/m) * np.sum(error)
        
        # 4. Update
        w = w - lr * gradient_w
        b = b - lr * gradient_b
        
        if epoch % 400 == 0:
            print(f"Epoch {epoch:4}: Loss = {loss:.4f}")
            
    return w, b, X_mean, X_std

def main():
    X, y = generate_classification_data()
    print("Training Logistic Regression (Fever Detector)...")
    
    w, b, x_mean, x_std = train_logistic_regression(X, y)
    
    # Test on a few samples
    test_temps = np.array([[36.0], [37.0], [38.0], [39.0]])
    # MUST normalize test data the same way as training data!
    test_temps_norm = (test_temps - x_mean) / x_std
    test_preds = sigmoid(test_temps_norm.dot(w) + b)
    
    print("\nFinal Testing:")
    for temp, pred in zip(test_temps, test_preds):
        result = "FEVER" if pred > 0.5 else "NORMAL"
        print(f"Temp: {temp[0]} | Confidence: {pred[0]:.4f} | Result: {result}")

if __name__ == "__main__":
    main()
