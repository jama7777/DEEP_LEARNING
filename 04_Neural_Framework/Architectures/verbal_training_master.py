import numpy as np
from master_stable_block import StableVerbalBlock

def train_on_paragraph():
    print("📖 VERBAL TRAINING MISSION: MEMORIZING A PARAGRAPH")
    print("=" * 65)

    # 1. THE DATA
    paragraph = "the sun rises in the east and sets in the west"
    words = paragraph.split()
    vocab = sorted(list(set(words)))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)
    
    # 2. THE EMBEDDING (Manual One-Hot for this demo)
    # We turn each word into a vector of size vocab_size
    def get_vector(word):
        vec = np.zeros((1, vocab_size))
        vec[0, word_to_id[word]] = 1.0
        return vec

    # 3. SETUP THE BLOCK
    # We want to take a word vector and output the SAME word vector (Auto-encoder)
    block = StableVerbalBlock(vocab_size, vocab_size)
    
    print(f"Vocab Size: {vocab_size} words")
    print(f"Training on: '{paragraph}'")
    print("-" * 65)

    # 4. THE TRAINING LOOP
    epochs = 1
    for epoch in range(epochs):
        total_loss = 0
        
        for word in words:
            # A. Forward Pass
            x = get_vector(word)
            pred = block.forward(x)
            
            # B. Calculate Loss (MSE)
            # Loss = (Pred - Target)^2
            error = pred - x
            loss = np.mean(error**2)
            total_loss += loss
            
            # C. THE INCOMING ERROR (dout)
            # This is the 'Baton Pass' from the Loss to the Layer
            # dLoss/dPred = 2 * (Pred - Target)
            dout = 2 * error / vocab_size
            
            # D. Backward Pass (Update Weights)
            block.backward(dout)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:3} | Loss: {total_loss:.6f} | Stability: {'Healthy ✅' if total_loss < 10 else 'Exploding ⚠️'}")

    # 5. THE FINAL TEST
    print("\n--- 🏁 FINAL RECONSTRUCTION TEST ---")
    correct = 0
    for word in words:
        x = get_vector(word)
        pred = block.forward(x)
        # Find the word the network 'thinks' it saw
        predicted_id = np.argmax(pred)
        predicted_word = vocab[predicted_id]
        
        status = "✅" if predicted_word == word else "❌"
        if status == "✅": correct += 1
        print(f"Target: '{word:<5}' | Pred: '{predicted_word:<5}' | {status}")

    print(f"\nAccuracy: {correct/len(words)*100:.1f}%")
    print("\n💡 THE DEEP TRUTH:")
    print("1. LayerNorm kept the word vectors from becoming 'too loud'.")
    print("2. Residuals allowed the error signal to reach the weights every time.")
    print("3. Adam steered the weights into the exact pattern of the paragraph.")

if __name__ == "__main__":
    train_on_paragraph()
