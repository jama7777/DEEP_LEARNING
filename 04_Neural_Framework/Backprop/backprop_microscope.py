import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def backprop_microscope():
    print("🔬 THE BACKPROPAGATION MICROSCOPE: THE CHAIN OF BLAME")
    print("=" * 65)

    # 1. SETUP
    x = 0.5        # Input (The Leverage)
    w = 0.8        # Weight (The Knob we are turning)
    target = 1.0   # Goal
    
    print(f"STORY: Input is {x}, Weight is {w}. Goal is to predict {target}.")
    
    # 2. FORWARD PASS (The Incident)
    z = x * w
    y = sigmoid(z)
    loss = 0.5 * (target - y)**2
    
    print("\n--- 🏗️ THE FORWARD PASS (The Crime) ---")
    print(f"1. Pre-Activation (z): {x} * {w} = {z:.4f}")
    print(f"2. Prediction (y): sigmoid({z:.4f}) = {y:.4f}")
    print(f"3. Total Loss: 0.5 * ({target} - {y:.4f})^2 = {loss:.4f}")

    print("\n--- 🕵️ THE BACKWARD PASS (The Investigation) ---")
    
    # Step A: The Direct Error
    # dL/dy = -(target - y)
    dl_dy = -(target - y)
    print(f"🔍 STEP 1: The Prediction Error (dL/dy)")
    print(f"   How far off were we? ({y:.4f} vs {target})")
    print(f"   'Blame' on the output: {dl_dy:.4f}")
    
    # Step B: The Gatekeeper (Activation Gradient)
    # dy/dz = sigmoid'(z)
    dy_dz = sigmoid_derivative(z)
    print(f"\n🔍 STEP 2: The Activation Gatekeeper (dy/dz)")
    print(f"   If we nudged the summation (z), how much would the prediction (y) change?")
    print(f"   Sigmoid slope at z={z:.4f} is {dy_dz:.4f}")
    
    # Step C: The Ripple to Summation (dL/dz)
    dl_dz = dl_dy * dy_dz
    print(f"\n🔍 STEP 3: The Combined Blame on Summation (dL/dz)")
    print(f"   {dl_dy:.4f} (Error) * {dy_dz:.4f} (Gatekeeper) = {dl_dz:.4f}")
    print(f"   This is the 'Error Signal' reaching the neuron's core.")

    # Step D: The Weight Leverage (dz/dw)
    # dz/dw = x
    dz_dw = x
    print(f"\n🔍 STEP 4: The Weight Leverage (dz/dw)")
    print(f"   How much does the Weight control the Summation?")
    print(f"   Since z = x * w, the derivative w.r.t weight is just the Input: {dz_dw:.4f}")
    print(f"   [POINT]: The input determines how much power the weight has to change the outcome.")

    # Step E: THE FINAL GRADIENT (dL/dw)
    dl_dw = dl_dz * dz_dw
    print(f"\n💎 THE FINAL VERDICT: Gradient for Weight (dL/dw)")
    print(f"   Formula: (Blame on Summation) * (Weight Leverage)")
    print(f"   {dl_dz:.4f} * {dz_dw:.4f} = {dl_dw:.4f}")

    print("\n" + "=" * 65)
    print("💡 DEEP INTUITION:")
    print(f"1. The Gradient ({dl_dw:.4f}) is negative. This means to LOWER loss, we must INCREASE weight.")
    print(f"2. If input (x) was 0, the gradient would be 0. (Dead leverage).")
    print(f"3. If we were already at target 1.0, the gradient would be 0. (No blame).")

if __name__ == "__main__":
    backprop_microscope()
