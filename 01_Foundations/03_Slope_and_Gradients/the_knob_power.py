import time

def compare_knob_power():
    print("--- THE KNOB POWER EXPERIMENT ---")
    print("Scenario: We have two knobs. One is 10,000x more powerful than the other.")
    print("The AI has to find the right volume (Target: 100) using BOTH.\n")
    
    target = 100.0
    
    # Knob A is powerful (Stadium Speaker)
    w_stadium = 0.0
    power_stadium = 100.0 
    
    # Knob B is weak (Tiny Headphone)
    w_headphone = 0.0
    power_headphone = 0.01
    
    print(f"{'Step':<5} | {'Stadium Slope':<15} | {'Headphone Slope':<15}")
    print("-" * 45)

    for i in range(1, 6):
        # Current Output from both (Total Sound)
        output_s = w_stadium * power_stadium
        output_h = w_headphone * power_headphone
        
        # SLOPES: How much does EACH knob affect the error?
        # Stadium Slope: 2 * (Error) * Power
        slope_s = 2 * (output_s - target) * power_stadium
        
        # Headphone Slope: 2 * (Error) * Power
        slope_h = 2 * (output_h - target) * power_headphone
        
        # The AI 'feels' the difference! 
        # It sees that the Stadium knob has a massive slope.
        print(f"{i:<5} | {slope_s:<15.2f} | {slope_h:<15.2f}")
        
        # Update weights (using a tiny learning rate)
        w_stadium -= 0.0001 * slope_s
        w_headphone -= 0.0001 * slope_h
        
        time.sleep(0.1)

    print("\nConclusion:")
    print(f"The Stadium Slope was {abs(slope_s/slope_h):.0f} times larger than the Headphone Slope.")
    print("The AI uses this difference to know: 'Turning the Stadium knob is high-risk, but turning the Headphone knob is low-risk.'")

if __name__ == "__main__":
    compare_knob_power()
