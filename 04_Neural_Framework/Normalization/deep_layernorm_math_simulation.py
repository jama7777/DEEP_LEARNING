import numpy as np

def simulate_deep_signal_propagation():
    print("🌊 DEEP SIGNAL PROPAGATION: THE MATH OF STABILITY")
    print("=" * 80)
    
    # Initial input (e.g., an embedding)
    x = np.array([0.5, -0.2, 0.1, 0.8])
    print(f"START: Input Vector x_0 = {x}")
    print(f"       Mean: {np.mean(x):.2f} | Std: {np.std(x):.2f}")
    print("-" * 80)

    # Simulation parameters
    n_layers = 5
    # We'll use random weights that slightly 'amplify' on average
    # to simulate how signals grow unstable in deep networks.
    np.random.seed(42)
    
    current_x_wild = x.copy()
    current_x_normed = x.copy()
    
    # LayerNorm Parameters (Learnable)
    gamma = 1.2
    beta = 0.5

    for layer in range(1, n_layers + 1):
        # 1. Weights: Random matrix (4x4)
        # We'll make weights slightly > 1 to show explosion
        W = np.random.randn(4, 4) * 1.5 
        
        print(f"LAYER {layer} PROCESSING...")
        
        # --- THE WILD PATH (NO NORM) ---
        current_x_wild = np.dot(W, current_x_wild)
        wild_mean = np.mean(current_x_wild)
        wild_std  = np.std(current_x_wild)
        
        # --- THE GOVERNED PATH (WITH LAYERNORM) ---
        # A. Linear Transform
        z = np.dot(W, current_x_normed)
        
        # B. LayerNorm Math (The 'Reset')
        mean_z = np.mean(z)
        std_z  = np.std(z)
        z_standard = (z - mean_z) / (std_z + 1e-8)
        
        # C. Re-scaling (The 'Freedom')
        current_x_normed = (z_standard * gamma) + beta
        
        norm_mean = np.mean(current_x_normed)
        norm_std  = np.std(current_x_normed)

        # SHOW THE MATH BATTLE
        print(f"   [WILD]   Vector: {current_x_wild}")
        print(f"            Stats: Mean={wild_mean:8.2f} | Std={wild_std:8.2f}  <-- GROWING!")
        
        print(f"   [NORMED] Vector: {current_x_normed}")
        print(f"            Stats: Mean={norm_mean:8.2f} | Std={norm_std:8.2f}  <-- STABLE (Beta/Gamma)")
        print("-" * 80)

    print("\n💡 FINAL ANALYSIS:")
    print(f"Wild Path End Std:   {np.std(current_x_wild):.2f} (Exploded!)")
    print(f"Normed Path End Std: {np.std(current_x_normed):.2f} (Controlled by Gamma)")
    print("\nWITHOUT LayerNorm, the signal either explodes to infinity (NaN)")
    print("or vanishes to zero. WITH LayerNorm, every layer starts with")
    print("fresh, predictable numbers.")

if __name__ == "__main__":
    simulate_deep_signal_propagation()
