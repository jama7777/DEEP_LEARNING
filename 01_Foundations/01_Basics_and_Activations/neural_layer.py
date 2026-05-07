import numpy as np

def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def main():
    # Let's see how different gatekeepers react to the same score
    scores = [5.0, 0.0, -5.0]
    
    print(f"{'Score':<8} | {'ReLU':<10} | {'Sigmoid':<10} | {'Tanh':<10}")
    print("-" * 45)
    
    for x in scores:
        r = relu(x)
        s = sigmoid(x)
        t = tanh(x)
        print(f"{x:<8.1f} | {r:<10.2f} | {s:<10.2f} | {t:<10.2f}")

    print("\nObservation:")
    print("- Notice how ReLU turns -5.0 into exactly 0.00 (It kills the signal).")
    print("- Notice how Tanh preserves the 'Negativeness' (-0.99).")
    print("- Notice how Sigmoid squishes everything to the 0-1 range.")

if __name__ == "__main__":
    main()
