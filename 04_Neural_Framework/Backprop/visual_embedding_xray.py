import numpy as np
import matplotlib.pyplot as plt

def visualize_embedding_drift():
    # 1. SETUP
    np.random.seed(42)
    sun = np.array([0.1, 2.0])  # Sun starts 'High'
    moon = np.array([5.0, 5.0]) # Moon starts 'Far'
    
    # Context target (Bright & Rises)
    target = np.array([2.5, 2.5])
    
    sun_history = [sun.copy()]
    moon_history = [moon.copy()]
    
    # 2. SIMULATE TRAINING (The Tug-of-War)
    lr = 0.1
    for i in range(20):
        # Both are pulled towards the same meaning
        sun += lr * (target - sun)
        moon += lr * (target - moon)
        
        sun_history.append(sun.copy())
        moon_history.append(moon.copy())
        
    sun_history = np.array(sun_history)
    moon_history = np.array(moon_history)

    # 3. PLOT THE JOURNEY
    plt.figure(figsize=(10, 6))
    plt.plot(sun_history[:, 0], sun_history[:, 1], 'ro-', label='Sun Journey')
    plt.plot(moon_history[:, 0], moon_history[:, 1], 'bo-', label='Moon Journey')
    plt.scatter(target[0], target[1], c='gold', s=200, marker='*', label='Shared Meaning (Bright/Rise)')
    
    plt.title("🌌 Embedding X-Ray: The Semantic Drift", fontsize=14)
    plt.xlabel("Dimension 1 (Brightness)")
    plt.ylabel("Dimension 2 (Time of Day)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the visualization
    save_path = "/Users/indra/Desktop/DEEP_LEARNING/embedding_drift_xray.png"
    plt.savefig(save_path)
    print(f"✅ Visual X-Ray saved to: {save_path}")

if __name__ == "__main__":
    visualize_embedding_drift()
