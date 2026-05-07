def calculate_slope(weight):
    # Let's use a simple Loss function: Loss = Weight^2
    # We want to find the slope at this specific 'weight'
    
    # 1. Current Loss
    loss_now = weight ** 2
    
    # 2. 'Probing': What if we increase weight by a tiny bit (0.001)?
    tiny_nudge = 0.001
    loss_after_nudge = (weight + tiny_nudge) ** 2
    
    # 3. Slope = (Change in Loss) / (Change in Weight)
    # This is 'Rise over Run'
    change_in_loss = loss_after_nudge - loss_now
    slope = change_in_loss / tiny_nudge
    
    return loss_now, slope

def main():
    print("--- THE SLOPE EXPLORER ---")
    print("Function: Loss = Weight^2\n")
    
    test_weights = [-10, -5, 0, 5, 10]
    
    print(f"{'Weight':<8} | {'Loss':<8} | {'Slope':<10} | {'Interpretation'}")
    print("-" * 60)
    
    for w in test_weights:
        loss, slope = calculate_slope(w)
        
        # Determine the interpretation
        if slope < -1: action = "Steep Downhill (Move Right!)"
        elif slope > 1: action = "Steep Uphill (Move Left!)"
        else: action = "Flat (The Bottom! 🎉)"
        
        print(f"{w:<8} | {loss:<8} | {slope:<10.2f} | {action}")

if __name__ == "__main__":
    main()
