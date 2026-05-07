import numpy as np

def deep_optimizer_logic():
    print("🧠 DEEP DIVE: WHY ADAM IS THE 'SMART DRIVER'")
    print("=" * 65)

    # Weights
    w1, w2 = 0.5, 0.5
    lr = 0.1
    
    # Optimizer Memory
    m1, v1 = 0.0, 0.0
    m2, v2 = 0.0, 0.0
    beta1, beta2 = 0.9, 0.999

    # SCENARIO:
    # Weight 1 has a STEADY gradient (Clean road)
    # Weight 2 has a JITTERY gradient (Bumpy road: +1 then -1)
    
    print("--- 🚦 STEP 1: The First Update ---")
    g1, g2 = 1.0, 1.0
    
    # Adam Update for W1
    m1 = beta1 * m1 + (1 - beta1) * g1
    v1 = beta2 * v1 + (1 - beta2) * (g1**2)
    w1 -= lr * (m1 / (np.sqrt(v1) + 1e-8))

    # Adam Update for W2
    m2 = beta1 * m2 + (1 - beta1) * g2
    v2 = beta2 * v2 + (1 - beta2) * (g2**2)
    w2 -= lr * (m2 / (np.sqrt(v2) + 1e-8))
    
    print(f"W1 (Steady) moved to: {w1:.4f}")
    print(f"W2 (Jitter) moved to: {w2:.4f}")

    print("\n--- 🚧 STEP 2: The Difference Appears ---")
    # Now, W1 stays steady, but W2 reverses direction (Jitters)
    g1, g2 = 1.0, -1.0 # W2 just jumped backward!
    
    # Adam Update W1
    m1 = beta1 * m1 + (1 - beta1) * g1
    v1 = beta2 * v1 + (1 - beta2) * (g1**2)
    w1 -= lr * (m1 / (np.sqrt(v1) + 1e-8))

    # Adam Update W2
    m2 = beta1 * m2 + (1 - beta1) * g2
    v2 = beta2 * v2 + (1 - beta2) * (g2**2)
    w2 -= lr * (m2 / (np.sqrt(v2) + 1e-8))

    print(f"W1 (Steady) now at: {w1:.4f} (Still moving strong)")
    print(f"W2 (Jitter) now at: {w2:.4f} (Adam slowed it down significantly!)")

    print("\n" + "=" * 65)
    print("💡 THE MAIN ADVANTAGE:")
    print("Adam realized W2 was 'jittery' because its gradient changed sign.")
    print("By looking at the Variance (v), it automatically 'Taps the Brakes'")
    print("on unstable weights while 'Stepping on the Gas' for stable ones.")
    print("\n[CONCLUSION]: This keeps the 1 Billion parameters in sync!")

if __name__ == "__main__":
    deep_optimizer_logic()
