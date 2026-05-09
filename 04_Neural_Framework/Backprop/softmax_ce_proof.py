import sympy as sp

def prove_miracle_math():
    print("🧪 THE SYMBOLIC PROOF: WHY DOES THE CALCULUS DISAPPEAR?")
    print("=" * 65)

    # 1. Define variables (z1, z2 are raw scores)
    z1, z2 = sp.symbols('z1 z2')
    
    # 2. Define Softmax for z1
    # P1 = e^z1 / (e^z1 + e^z2)
    p1 = sp.exp(z1) / (sp.exp(z1) + sp.exp(z2))
    
    # 3. Define Cross-Entropy Loss (if z1 is the correct answer)
    # Loss = -log(P1)
    loss = -sp.log(p1)
    
    print("THE FORMULAS:")
    print(f"Softmax P1: {p1}")
    print(f"Loss:       {loss}")
    
    # 4. THE CALCULUS STEP
    # We take the derivative of Loss with respect to z1
    derivative = sp.diff(loss, z1)
    
    print("\n" + "-" * 65)
    print("THE RAW DERIVATIVE (Before simplifying):")
    print(derivative)
    
    # 5. THE SIMPLIFICATION
    # We ask Sympy to clean up the mess
    simple_derivative = sp.simplify(derivative)
    
    print("\n" + "-" * 65)
    print("THE SIMPLIFIED DERIVATIVE (The Miracle):")
    print(simple_derivative)
    
    print("\n" + "-" * 65)
    print("💡 THE DEEP TRUTH:")
    print("Look at the simplified result! It is:")
    print("      (e^z1 / (e^z1 + e^z2)) - 1")
    print("\nWhich is exactly:")
    print("      Probability_1 - 1.0 (Target)")
    
    print("\nConclusion: The derivative DID NOT disappear.")
    print("The Exponential (Softmax) and the Logarithm (Cross-Entropy)")
    print("are mathematical opposites. When they meet in calculus,")
    print("they erase each other's complexity, leaving only the 'Gap'.")

if __name__ == "__main__":
    prove_miracle_math()
