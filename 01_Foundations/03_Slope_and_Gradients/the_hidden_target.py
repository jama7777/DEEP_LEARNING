import time

def discover_hidden_weight():
    # THE SECRET: The universe has a hidden rule: Answer = Weight * 7.25
    # The AI does NOT know the 7.25. It only knows the answer should be 100.
    
    target_answer = 100.0
    weight = 0.0 # Start at zero
    learning_rate = 0.01
    
    print("--- THE HIDDEN TARGET MISSION ---")
    print(f"Goal: Make the output {target_answer}")
    print("AI Knowledge: It knows the target is 100, but it DOESN'T know how the knob works.\n")
    print(f"{'Step':<5} | {'Weight':<10} | {'Output':<10} | {'Slope':<10} | {'Discovery'}")
    print("-" * 65)

    for i in range(1, 11):
        # 1. Forward Pass: See what happens with current weight
        # In reality, this is: Output = Weight * Hidden_Factor
        output = weight * 7.25
        
        # 2. Calculate Loss (Distance from 100)
        loss = (output - target_answer) ** 2
        
        # 3. Calculate Slope (The Discovery)
        # We check: 'If I change weight by 1, how much does the output change?'
        # The slope will literally REVEAL the hidden 7.25!
        # Math: Derivative of (W*7.25 - 100)^2 is 2 * (W*7.25 - 100) * 7.25
        slope = 2 * (output - target_answer) * 7.25
        
        # 4. Update
        weight -= learning_rate * slope
        
        discovery = "Learning Knob Power..."
        if i > 5: discovery = "Found the hidden rule! 🔍"

        print(f"{i:<5} | {weight:<10.2f} | {output:<10.2f} | {slope:<10.2f} | {discovery}")
        time.sleep(0.1)

    print("\nDeep Conclusion:")
    print(f"The AI found the weight should be {weight:.2f}")
    print(f"Proof: {weight:.2f} * 7.25 = {weight * 7.25:.2f}")
    print("The Slope was the 'Spy' that went inside the system and found the hidden multiplier.")

if __name__ == "__main__":
    discover_hidden_weight()
