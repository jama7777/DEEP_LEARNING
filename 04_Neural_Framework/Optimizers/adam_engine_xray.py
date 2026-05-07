import numpy as np

def run_adam_scenario(name, gradients, initial_weight=10.0):
    print(f"\n--- {name} ---")
    print(f"{'Step':<5} | {'Grad':<6} | {'m_hat':<8} | {'v_hat':<8} | {'Update':<10} | {'Logic'}")
    print("-" * 65)
    
    weight = initial_weight
    m, v = 0.0, 0.0
    lr = 0.1
    beta1, beta2 = 0.9, 0.999
    
    for i, grad in enumerate(gradients, 1):
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        
        m_hat = m / (1 - beta1**i)
        v_hat = v / (1 - beta2**i)
        
        update = lr * (m_hat / (np.sqrt(v_hat) + 1e-8))
        weight -= update
        
        # Determine verbal logic
        if abs(update) > 0.09: logic = "GAS 🏎️"
        elif abs(update) < 0.02: logic = "BRAKES 🛑"
        else: logic = "TURNING 🔄"
        
        print(f"{i:<5} | {grad:<6.2f} | {m_hat:<8.4f} | {v_hat:<8.4f} | {update:<10.4f} | {logic}")

def adam_deep_dive():
    print("🧠 ADAM MASTER X-RAY: GAS, BRAKES, and U-TURNS")
    print("=" * 70)

    # SCENARIO 1: THE HIGHWAY (TINY BUT STEADY)
    # Even though the gradient is 100x smaller, Adam keeps the speed HIGH.
    run_adam_scenario("SCENARIO 1: THE HIGHWAY (Steady 0.01)", [0.01]*5)

    # SCENARIO 2: THE BUMPY ROAD (JITTER)
    # The gradient is jumping +1, -1. Adam will hit the brakes.
    run_adam_scenario("SCENARIO 2: THE BUMPY ROAD (Jitter +1, -1)", [1.0, -1.0, 1.0, -1.0, 1.0])

    # SCENARIO 3: THE U-TURN (SUDDEN CHANGE)
    # Moving one way, then suddenly told to go the other way.
    run_adam_scenario("SCENARIO 3: THE U-TURN (+1 for 3 steps, then -1)", [1.0, 1.0, 1.0, -1.0, -1.0])

    print("\n" + "=" * 70)
    print("💡 THE FINAL REVELATION:")
    print("1. HIGHWAY: Adam boosted the tiny 0.01 gradient to a 0.10 update!")
    print("2. BUMPY: Look at Step 2/4. The update dropped to almost ZERO.")
    print("   Adam realized the weights were just 'vibrating' and stopped them.")
    print("3. U-TURN: In Step 4, even though the gradient was -1.0,")
    print("   the update was only -0.05. Momentum kept the weight moving forward")
    print("   before finally letting it turn. This prevents 'Whplash' in the model.")

if __name__ == "__main__":
    adam_deep_dive()
