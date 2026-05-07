import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    # 1. MULTIPLE INPUTS (Speed, Strength, Endurance)
    # Let's say this athlete is: Fast (0.9), Strong (0.8), but Low Endurance (0.2)
    inputs = np.array([0.9, 0.8, 0.2])
    target = 1.0  # We want to train the AI to say YES (Elite)
    
    # 2. MULTIPLE WEIGHTS (One for each input)
    # We start with random weights for each trait
    weights = np.array([0.1, 0.2, -0.1])
    bias = 0.0
    lr = 0.5
    
    print("--- Training a Multi-Input Brain ---")
    print(f"Inputs: {inputs}")
    print(f"Initial Weights: {weights}\n")

    for epoch in range(5):
        # 3. THE DOT PRODUCT (The shortcut for many inputs)
        # Instead of w1*x1 + w2*x2 + w3*x3, we do one 'dot' operation
        score = np.dot(inputs, weights) + bias
        print("score " , score)
        prediction = sigmoid(score)
        print("prediction " , prediction)
        error = prediction - target
        print("error " , error)

        loss = error ** 2
        print("loss " , loss)
        
        # 4. MULTIPLE GRADIENTS
        # Because we have 3 weights, we need 3 gradients!
        # The math is the same: 2 * error * (sigmoid_derivative) * input
        common_signal = 2 * error * (prediction * (1 - prediction))
        print("common_signal " , common_signal)
        
        # This one line updates ALL weights at once using the inputs vector!
        weights_gradients = common_signal * inputs
        print("weights_gradients " , weights_gradients)
        bias_gradient = common_signal
        print("bias_gradient " , bias_gradient)
        print("learning rate " , lr)
        
        # Update everything
        weights -= lr * weights_gradients
        print("weights " , weights)
        bias -= lr * bias_gradient
        print("bias " , bias)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f} | Prediction = {prediction:.4f}")

    print(f"\nFinal Prediction: {prediction:.4f}")
    print(f"Final Weights: {weights}")
    print("Notice how the weights for Speed and Strength increased, but Endurance didn't change as much!")

if __name__ == "__main__":
    main()
