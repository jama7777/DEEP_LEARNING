import numpy as np

def experiment_gradient_survival():
    print("🧪 EXPERIMENT: THE GRADIENT SURVIVAL TEST")
    print("=" * 60)
    
    # 1. SETUP: 50 Layers
    num_layers = 50
    incoming_error = 1.0 # The CEO is very angry!
    
    print(f"CEO Complaint (Initial Error): {incoming_error}")
    
    # --- SCENARIO A: THE SWAMP (Standard Layers) ---
    # Every layer has small weights (0.9). We multiply by weights every time.
    swamp_error = incoming_error
    weight_scale = 0.9 
    for i in range(num_layers):
        swamp_error *= weight_scale
        
    # --- SCENARIO B: THE HIGHWAY (Residual Layers) ---
    # Every layer adds the original error back (+1 path).
    highway_error = incoming_error
    # In Residual, the gradient is: dout + (dout * weights)
    # Even if weights are small, we ALWAYS keep the 1.0 from the skip path.
    for i in range(num_layers):
        # Grad = Grad_from_above * (1 + layer_effect)
        highway_error = highway_error * (1 + 0.001) # 0.001 is the tiny layer change
        
    print(f"\n--- AFTER {num_layers} LAYERS ---")
    print(f"💀 SWAMP ERROR (Standard): {swamp_error:.10f}")
    print(f"🚀 HIGHWAY ERROR (Residual): {highway_error:.4f}")
    
    print("\n💡 THE DEEP TRUTH:")
    print("1. In the SWAMP, the error 'Vanished'. Layer 1 hears almost NOTHING.")
    print("2. On the HIGHWAY, the error stayed LOUD and clear.")
    print("3. RESULT: Layer 1 can start learning IMMEDIATELY. This is why it's faster!")

if __name__ == "__main__":
    experiment_gradient_survival()
