import numpy as np
from master_stable_block import StableVerbalBlock

def deep_inside_error_math_xray():
    print("🔬 THE DEEP INSIDE X-RAY: HOW DOUT CHANGES THE SOUL OF THE MODEL")
    print("=" * 75)

    # 1. SETUP
    vocab = ["the", "sun", "rises", "in", "east", "and", "sets", "west"]
    vocab_size = len(vocab)
    word_to_id = {w: i for i, w in enumerate(vocab)}
    
    # Word: 'sun' (ID: 1)
    target = np.zeros((1, vocab_size))
    target[0, word_to_id["sun"]] = 1.0
    
    block = StableVerbalBlock(vocab_size, vocab_size)
    
    # FORWARD PASS (To get values in cache)
    prediction = block.forward(target)
    
    # THE INCOMING ERROR (The Baton)
    error = prediction - target
    dout = 2 * error / vocab_size
    
    print(f"1. THE BATON RECEIVED (dout):\n{dout}")
    print("-" * 75)

    # 2. INSIDE STEP 1: CHANGING THE WEIGHTS (dW)
    # dW = x.T * dout
    # This shows which weights are to blame.
    x_scaled = block.cache['x_scaled']
    dW = np.dot(x_scaled.T, dout)
    
    print("2. INSIDE THE WEIGHTS (dW):")
    print("This matrix tells us how to change the 'Thinking' of the layer.")
    print(f"dW Slice (First 2 rows):\n{dW[:2, :]}")
    print("\n💡 DEEP POINT: Notice how ONLY the weights connected to")
    print("the active input neurons have large dW values. The error")
    print("is 'hunting' for the weights that caused the mistake!")

    # 3. INSIDE STEP 2: THE BATON TRANSLATION (dx)
    # dx = dout * W.T
    # This is what we pass to the layer BEFORE us.
    dx_thinking = np.dot(dout, block.W.T)
    
    print("\n" + "-" * 75)
    print("3. THE BATON TRANSLATION (dx_thinking):")
    print(f"Translated Error for previous layer: {dx_thinking}")
    print("\n💡 DEEP POINT: This is the error signal translated into")
    print("the language of the PREVIOUS layer. We multiplied by W.T")
    print("to 'undo' the thinking and find the source.")

    # 4. INSIDE STEP 3: THE RESIDUAL RESCUE
    # In Residual: dx_total = dx_thinking + dout
    dx_total = dx_thinking + dout
    
    print("\n" + "-" * 75)
    print("4. THE RESIDUAL RESCUE (dx_total):")
    print(f"Final Baton Passed Back: {dx_total}")
    print("\n💡 THE SUPREME TRUTH:")
    print("Because we ADD the incoming error (dout) back to the translated error,")
    print("the signal is REINFORCED. Even if the weights (W) were zero,")
    print("the error would still reach the previous layer perfectly.")

if __name__ == "__main__":
    deep_inside_error_math_xray()
