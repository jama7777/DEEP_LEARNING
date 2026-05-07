import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    """Create some linear-looking data with noise."""
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y

def train_linear_regression(X, y, lr=0.1, epochs=5):
    """Implement simple gradient descent for linear regression."""
    m = len(X)
    # Initialize parameters (weights and bias)
    theta = np.random.randn(2, 1)
    
    # Add bias term (x0 = 1) to each instance
    X_b = np.c_[np.ones((m, 1)), X]
    
    for epoch in range(epochs):
        # Calculate loss for monitoring
        predictions = X_b.dot(theta)
        print("predictions: " , predictions)
        loss = np.mean((predictions - y)**2)
        print("loss: " , loss)
        # Calculate gradients and update
        gradients = 2/m * X_b.T.dot(predictions - y)
        print("gradients: " , gradients)
        theta = theta - lr * gradients
        print("theta: " , theta)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3}: Bias = {theta[0][0]:.4f} | Weight = {theta[1][0]:.4f} | Loss = {loss:.4f}")
        
    return theta , gradients

def main():
    X, y = generate_data()
    print("Training Linear Regression from scratch...")
    print(X)
    print(y)
    theta , gradients = train_linear_regression(X, y)
    print(f"Learned parameters:\nBias: {theta[0][0]:.4f} (True value: 4)\nWeight: {theta[1][0]:.4f} (True value: 3)")
    print(f"Gradients: {gradients}")
    # Plotting (Optional - depends if GUI is available)
    # plt.scatter(X, y)
    # plt.plot(X, np.c_[np.ones((len(X), 1)), X].dot(theta), color='red')
    # plt.show()

if __name__ == "__main__":
    main()
