import numpy as np

class ResidualBlock:
    def __init__(self, layer):
        self.layer = layer

    def forward(self, x):
        # Save the original input (The 'Skip' path)
        self.identity = x
        
        # Process through the layer (The 'Learn' path)
        out = self.layer.forward(x)
        
        # ADD them together
        return x + out

    def backward(self, dout):
        # 1. Gradient from the 'Learn' path
        d_layer = self.layer.backward(dout)
        
        # 2. Gradient from the 'Skip' path is just 1.0 * dout
        d_skip = dout
        
        # 3. Combine them
        return d_layer + d_skip

def test_superhighway():
    print("🛣️ THE GRADIENT SUPERHIGHWAY TEST")
    print("=" * 60)

    # Imagine a layer that is 'broken' and outputs almost 0
    class BrokenLayer:
        def forward(self, x): return x * 0.0001
        def backward(self, dout): return dout * 0.0001

    x = np.array([10.0, 20.0, 30.0])
    res_block = ResidualBlock(BrokenLayer())

    # Forward
    output = res_block.forward(x)
    print(f"Input:  {x}")
    print(f"Output: {output} (Notice it stayed close to the input!)")

    # Backward
    dout = np.array([1.0, 1.0, 1.0]) # Incoming error signal
    dx = res_block.backward(dout)
    
    print(f"\nIncoming Gradient: {dout}")
    print(f"Resulting Gradient: {dx}")
    print("\n💡 DEEP POINT:")
    print("Even though the layer was 'broken' (0.0001), the gradient survived")
    print("because of the Skip Connection (it's nearly 1.0)! This is how")
    print("very deep models like GPT keep their gradients healthy.")

if __name__ == "__main__":
    test_superhighway()
