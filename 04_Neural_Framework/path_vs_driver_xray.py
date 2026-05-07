import numpy as np

def path_vs_driver_xray():
    print("🛣️ SIGNAL vs. DRIVER: RELU vs. ADAM")
    print("=" * 60)

    # 1. THE SIGNAL (The Road)
    # Sigmoid 'squashes' the blame. ReLU keeps it 'pure'.
    sigmoid_grad = 0.0001 # Faded signal
    relu_grad = 1.0      # Strong, living signal
    
    # 2. THE DRIVER (Adam's Memory)
    m_sigmoid, v_sigmoid = 0.0, 0.0
    m_relu, v_relu = 0.0, 0.0
    lr = 0.1
    beta1, beta2 = 0.9, 0.999

    print("--- 🌫️ SCENARIO 1: SQUASHED ROAD (Sigmoid) ---")
    # Even with Adam, if the signal is dead, nothing happens.
    m_sigmoid = beta1 * m_sigmoid + (1 - beta1) * sigmoid_grad
    v_sigmoid = beta2 * v_sigmoid + (1 - beta2) * (sigmoid_grad**2)
    update_sigmoid = lr * (m_sigmoid / (np.sqrt(v_sigmoid) + 1e-8))
    
    print(f"Signal Strength: {sigmoid_grad}")
    print(f"Weight Update:   {update_sigmoid:.8f}")
    print("Result: The model is 'Deaf'. Adam has no message to listen to.")

    print("\n--- 🚀 SCENARIO 2: CLEAR ROAD (ReLU) ---")
    # With a loud signal, Adam can finally 'Drive'.
    m_relu = beta1 * m_relu + (1 - beta1) * relu_grad
    v_relu = beta2 * v_relu + (1 - beta2) * (relu_grad**2)
    update_relu = lr * (m_relu / (np.sqrt(v_relu) + 1e-8))
    
    print(f"Signal Strength: {relu_grad}")
    print(f"Weight Update:   {update_relu:.4f}")
    print("Result: The signal is loud! Adam uses it to move the model fast.")

    print("\n" + "=" * 60)
    print("💡 THE DEEP TRUTH:")
    print("1. ReLU is the FIBER OPTIC CABLE. It keeps the data perfect.")
    print("2. Adam is the SYSTEM ADMIN. He decides what to do with the data.")
    print("\n[CONCLUSION]: ReLU keeps the gradients 'alive',")
    print("and Adam makes the learning 'smart'. You need BOTH.")

if __name__ == "__main__":
    path_vs_driver_xray()
