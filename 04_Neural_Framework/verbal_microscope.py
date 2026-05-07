import numpy as np

def summation_bucket_trace(inputs, weights, col_idx):
    print(f"\n--- 🗑️ THE SUMMATION BUCKET (Column {col_idx}) ---")
    print("Every multiplication is an 'Argument'. We add them all to get the 'Verdict'.")
    
    bucket_total = 0
    for i in range(len(inputs)):
        val = inputs[i]
        weight = weights[i, col_idx]
        product = val * weight
        
        # Action
        old_total = bucket_total
        bucket_total += product
        
        # Visualization
        arrow = "🟢" if product > 0 else "🔴" if product < 0 else "⚪"
        print(f"  {arrow} Argument {i}: {val:+.2f} * {weight:+.2f} = {product:+.2f} | Bucket: {old_total:+.2f} -> {bucket_total:+.2f}")
    
    print(f"\n🏁 FINAL VERDICT: {bucket_total:.2f}")
    return bucket_total

def main():
    # Context (8 numbers)
    context = np.array([0.1, 0.2, -0.1, 0.0, -0.2, 0.1, 0.3, -0.1])
    
    # Weights for 3 words (8x3)
    # Let's give 'ai' some complex weights to see the bucket work
    weights = np.zeros((8, 3))
    weights[0, 2] = 1.0   # i says 'Yes'
    weights[2, 2] = -0.5  # i says 'Maybe No'
    weights[4, 2] = 1.0   # love says 'Yes'
    weights[6, 2] = -1.0  # love says 'No!'

    print("🚀 THE DOT PRODUCT MICROSCOPE: HOW 8 BECOMES 1")
    print("="*60)

    # Trace the calculation for 'ai'
    final_verdict = summation_bucket_trace(context, weights, 2)

    print("\n[DEEP POINT]: Notice how the bucket keeps accumulating.")
    print("Positive arguments (+0.10) and negative arguments (-0.20, -0.30) are fighting.")
    print("The weights ARE the arguments. The dot product is the TOTAL DEBATE.")

if __name__ == "__main__":
    main()
