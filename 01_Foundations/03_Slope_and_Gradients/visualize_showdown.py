import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0, else 0."""
    return (z > 0).astype(float)

def visualize_training():
    # Initial Conditions
    x, target, lr =  5, 0, 0.1
    w1, w2, w3 = 0.1, 0.1, 0.1
    b1, b2, b3 = 0.0, 0.0, 0.0
    p1, p2, p3 = 0.0, 0.0, 0.0
    history_w1, history_w2, history_w3 = [], [], []
    history_b1, history_b2, history_b3 = [], [], []
    history_p1, history_p2, history_p3 = [], [], []
    steps = 1000
    
    for step in range(steps):
        history_w1.append(w1)
        history_w2.append(w2)
        history_w3.append(w3)

        history_b1.append(b1)
        history_b2.append(b2)
        history_b3.append(b3)

        history_p1.append(p1)
        history_p2.append(p2)
        history_p3.append(p3)

        p1, p2, p3 = x*w1+b1, x*w2+b2, x*w3+b3

        s1 = sigmoid(p1)
        s2 = sigmoid(p2)
        s3 = sigmoid(p3)
        
        # Gradients (Comparing 3 Strategies)
        # S1: Only p (Confidence)
        grad_1 = 2 * (s1 - target) * s1

        # S2: Only (1-p) (Doubt)
        grad_2 = 2 * (s2 - target) * (1 - s2)

        # S3: p*(1-p) (The Balanced Way)
        grad_3 = 2 * (s3 - target) * s3 * (1 - s3)

        b1 = b1 - lr * grad_1
        b2 = b2 - lr * grad_2
        b3 = b3 - lr * grad_3

        w1 -= lr * grad_1 * x
        w2 -= lr * grad_2 * x
        w3 -= lr * grad_3 * x
    
    # Set up the style
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    # Plotting
    plt.plot(history_w1, label='S1: Only p (Aggressive)', color='#00d4ff', linewidth=4, alpha=0.8)
    plt.plot(history_w2, label='S2: Only (1-p) (Cautious)', color='#ff007f', linewidth=3, alpha=0.8)
    plt.plot(history_w3, label='S3: p*(1-p) (The Balanced Way)', color='#39ff14', linewidth=2.5, alpha=0.8)
    plt.plot()
    # Aesthetics
    plt.title('Training Showdown: Sensitivity Strategies', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Weight Value (W)', fontsize=14)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.legend(fontsize=12, frameon=True, shadow=True, facecolor='#111111')
    
    # Save the plot
    save_path = '/Users/indra/Desktop/DEEP_LEARNING/01_Foundations/03_Slope_and_Gradients/showdown_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {save_path}")


    plt.figure(figsize=(10, 6))
    
    # Plotting
    plt.plot(history_b1, label='S1: Only p (Aggressive)', color='#00d4ff', linewidth=2.0, alpha=0.8)
    plt.plot(history_b2, label='S2: Only (1-p) (Cautious)', color='#ff007f', linewidth=3, alpha=0.8)
    plt.plot(history_b3, label='S3: p*(1-p) (The Balanced Way)', color='#39ff14', linewidth=2.5, alpha=0.8)
    plt.plot()
    # Aesthetics
    plt.title('Training Showdown: Sensitivity Strategies', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Weight Value (W)', fontsize=14)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.legend(fontsize=12, frameon=True, shadow=True, facecolor='#111111')
    
    # Save the plot
    save_path = '/Users/indra/Desktop/DEEP_LEARNING/01_Foundations/03_Slope_and_Gradients/showdown_bias.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {save_path}")

    plt.figure(figsize=(10, 6))
    
    # Plotting
    plt.plot(history_p1, label='S1: Only p (Aggressive)', color='#00d4ff', linewidth=2.0, alpha=0.8)
    plt.plot(history_p2, label='S2: Only (1-p) (Cautious)', color='#ff007f', linewidth=3, alpha=0.8)
    plt.plot(history_p3, label='S3: p*(1-p) (The Balanced Way)', color='#39ff14', linewidth=2.5, alpha=0.8)
    
    # Aesthetics
    plt.title('Training Showdown: Sensitivity Strategies', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Prediction Value (P)', fontsize=14)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.legend(fontsize=12, frameon=True, shadow=True, facecolor='#111111')
    
    # Save the plot
    save_path = '/Users/indra/Desktop/DEEP_LEARNING/01_Foundations/03_Slope_and_Gradients/showdown_pred.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {save_path}")

if __name__ == "__main__":
    visualize_training()
