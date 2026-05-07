import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    # 1. THE INPUT MATRIX (3 Athletes, 3 Features each)
    # Rows = [Speed, Strength, Endurance]
    # Athlete 1: [0.9, 0.8, 0.2] -> Should be ELITE (1)
    # Athlete 2: [0.1, 0.2, 0.9] -> Should NOT be Elite (0)
    # Athlete 3: [0.4, 0.5, 0.4] -> Should NOT be Elite (0)
    
    X = np.array([
        [0.9, 0.8, 0.2],
        [0.1, 0.2, 0.9],
        [0.4, 0.5, 0.4]
    ])
    
    y = np.array([1.0, 0.0, 0.0]) # The targets for each athlete
    
    # 2. INITIAL WEIGHTS & BIAS
    weights = np.random.randn(3)
    bias = 0.0
    lr = 0.5
    
    print("--- Training a BATCH BRAIN ---")
    print(f"Initial Weights: {weights}\n")

    for epoch in range(5):
        # 3. MATRIX MULTIPLICATION (The Super Shortcut)
        # np.dot(X, weights) calculates scores for ALL 3 athletes at once!
        scores = np.dot(X, weights) + bias
        print("scores: " , scores)
        predictions = sigmoid(scores)
        print("predictions: " , predictions)
        # 4. BATCH ERROR
        # error is now a vector of 3 differences
        errors = predictions - y
        print("errors: " , errors)
        loss = np.mean(errors ** 2)
        print("loss: " , loss)
        # 5. BATCH GRADIENTS (Averaging the responsibility)
        # We find the error signal for each athlete
        common_signals = 2 * errors * (predictions * (1 - predictions))
        print("common_signals: " , common_signals)
        
        # Matrix math trick: We multiply the entire Matrix by the error signals
        # and take the average across the 3 athletes.
        weights_gradients = np.dot(X.T, common_signals) / 3
        print("weights_gradients: " , weights_gradients)
        bias_gradient = np.mean(common_signals)
        print("bias_gradient: " , bias_gradient)
        
        # Update everything
        weights -= lr * weights_gradients
        print("weights: " , weights)
        bias -= lr * bias_gradient
        print("bias: " , bias)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4}: Loss = {loss:.6f} | Predictions = {predictions}")

    print("\nFinal Results:")
    for i, pred in enumerate(predictions):
        result = "ELITE" if pred > 0.5 else "NOT ELITE"
        print(f"Athlete {i+1}: Prediction = {pred:.4f} -> {result}")

    print(f"\nFinal Weights: {weights}")
    print("The AI now knows exactly which 'mixture' of traits makes an elite athlete!")

if __name__ == "__main__":
    main()
