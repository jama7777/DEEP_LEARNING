import numpy as np

def detailed_batch_math():
    print("🔬 DEEP DETAIL: 1 WEIGHT vs. 3 PEOPLE")
    print("=" * 60)

    # 1. THE SINGLE KNOB (The Weight)
    weight = 0.5
    lr = 0.1
    
    # 2. THE BATCH (3 People with different goals)
    # Person 1 wants output 1.0 (Higher)
    # Person 2 wants output 0.0 (Lower)
    # Person 3 wants output 0.6 (Slightly Higher)
    inputs =  np.array([1.0, 1.0, 1.0])
    targets = np.array([1.0, 0.0, 0.6])

    print(f"Current Weight: {weight}")
    print("\n--- 🏃 STEP 1: PARALLEL FORWARD PASS ---")
    preds = inputs * weight
    print(f"Predictions for Batch: {preds}")

    print("\n--- 🕵️ STEP 2: INDIVIDUAL COMPLAINTS (Gradients) ---")
    # Error = Pred - Target
    # Gradient = Error * Input
    errors = preds - targets
    grads = errors * inputs # Individual blame
    
    for i in range(3):
        direction = "UP ⬆️" if grads[i] < 0 else "DOWN ⬇️"
        print(f"Person {i+1}: Goal {targets[i]} | Error {errors[i]:.2f} | Vote: {direction} ({grads[i]:.2f})")

    print("\n--- ⚖️ STEP 3: THE BATCH ACCUMULATION (The Bucket) ---")
    # This is what happens in np.dot(input.T, error)
    total_blame = np.sum(grads)
    average_blame = np.mean(grads)
    
    print(f"Sum of all votes: {total_blame:.2f}")
    print(f"Average (The Consensus): {average_blame:.2f}")

    print("\n--- 💎 STEP 4: THE SINGLE UPDATE ---")
    new_weight = weight - (lr * average_blame)
    print(f"New Weight = {weight} - ({lr} * {average_blame:.2f}) = {new_weight:.4f}")

    print("\n" + "=" * 60)
    print("💡 THE VERBAL CONCLUSION:")
    print("We didn't 'split' the weight. We just found one number")
    print("that makes the TOTAL error for all 3 people as small as possible.")
    print("The weight moved slightly toward 0.6 to satisfy the majority.")

if __name__ == "__main__":
    detailed_batch_math()
