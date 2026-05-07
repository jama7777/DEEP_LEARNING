import numpy as np
from xray_utils import show_detailed_math

class VerbalBackpropXRay:
    def __init__(self):
        # 1. THE DICTIONARY (Embedding Weights)
        # 5 words, 4 dimensions each
        self.emb_weights = np.array([
            [ 0.1,  0.2, -0.1,  0.0], # Index 0: "i"
            [-0.2,  0.1,  0.3, -0.1], # Index 1: "love"
            [ 0.0,  0.0,  0.1,  0.1], # Index 2: "ai"
            [ 0.1, -0.1,  0.0,  0.2], # Index 3: "is"
            [-0.1,  0.2,  0.1,  0.0]  # Index 4: "deep"
        ])
        
        # 2. THE BRAIN (Dense Weights)
        self.dense_weights = np.zeros((8, 3))
        # Initializing with a tiny bit of knowledge
        self.dense_weights[0, 2] = 0.5  
        self.dense_weights[4, 2] = 0.5  
        self.dense_biases = np.zeros((1, 3))

    def train(self, epochs=3, lr=0.5):
        print(f"🚀 STARTING TRAINING FOR {epochs} EPOCHS (LR={lr})")
        print("Goal: Make the network predict 'ai' after 'i love'.")
        
        words = ["i", "love", "ai"]
        
        for epoch in range(1, epochs + 1):
            print("\n" + "="*80)
            print(f"🌟 EPOCH {epoch}")
            print("="*80)

            # --- 🏗️ FORWARD PASS ---
            # Input: "i" (0) and "love" (1)
            idx_i = 0
            idx_love = 1
            
            v_i = self.emb_weights[idx_i:idx_i+1]
            v_love = self.emb_weights[idx_love:idx_love+1]
            combined = np.concatenate([v_i, v_love], axis=1)
            scores = np.dot(combined, self.dense_weights) + self.dense_biases
            
            if epoch < 6:
                print("\n" + "="*40)
                print("🏗️ FORWARD DEEP ZOOM: THE WORD PREDICTION")
                print("="*40)
                print("\n🔍 STEP 1: Processing the 8D Sentence Signal")
                print("Formula: Combined @ Dense_Weights")
                show_detailed_math("Scores (1x3 Matrix)", [combined, self.dense_weights], scores, operation="*")
                
                print("\n" + "-"*40)
                print("🧠 ELEMENT-BY-ELEMENT: HOW WE GET 'ai'")
                print("-"*40)
                ai_weights = self.dense_weights[:, 2] 
                contributions = combined[0] * ai_weights
                for i, val in enumerate(contributions):
                    word = "'i'" if i < 4 else "'love'"
                    print(f"  Feature {i} from {word}: {combined[0,i]:+.2f} * {ai_weights[i]:+.2f} = {val:+.2f} vote")
                print(f"\n🏁 TOTAL VOTE FOR 'ai': {np.sum(contributions):.4f}")
                print("="*40)

            print(f"PREDICTION BEFORE UPDATE: {scores[0]}")
            print(f"CURRENT WINNER: {words[np.argmax(scores)]}")

            # --- 🎯 ERROR & TARGET ---
            target = np.array([[0.0, 0.0, 1.0]]) # Target is 'ai'
            dL_dZ = scores - target
            
            # --- 🕵️ BACKWARD PASS (X-Ray for Epoch 1) ---
            if epoch < 3:
                print("\n[EPOCH 1 DETAIL: The first nudge]")
                show_detailed_math("The Error Signal (dL/dZ)", [scores, target], dL_dZ, operation="-")
            
            # 1. Gradients
            dL_dW_dense = np.dot(combined.T, dL_dZ)
            dL_dCombined = np.dot(dL_dZ, self.dense_weights.T)
            
            if epoch < 6:
                print("\n" + "="*40)
                print("🕵️ BACKWARD DEEP ZOOM: THE MATRIX REVERSE")
                print("="*40)
                print("\n🔍 STEP 1: Creating the Brain's Blame Grid (dL/dW)")
                print("Formula: Combined.T @ Error")
                show_detailed_math("dL/dW_dense (8x3 Matrix)", [combined.T, dL_dZ], dL_dW_dense, operation="*")

                print("\n🔍 STEP 2: Sending Warning back to Words (dL/dCombined)")
                print("Formula: Error @ Dense_Weights.T")
                show_detailed_math("dL/dCombined (1x8 Vector)", [dL_dZ, self.dense_weights.T], dL_dCombined, operation="*")
                print("="*40)

            dL_vi = dL_dCombined[:, :4]
            dL_vlove = dL_dCombined[:, 4:]

            # 2. UPDATE (The actual learning)
            self.dense_weights -= lr * dL_dW_dense
            self.dense_biases -= lr * np.sum(dL_dZ, axis=0, keepdims=True)
            self.emb_weights[idx_i] -= lr * dL_vi[0]
            self.emb_weights[idx_love] -= lr * dL_vlove[0]

            if epoch == 1:
                print("\n🔍 ACTION: Updating weights with LR=0.5")
                print("The 'Brain' weights and 'Dictionary' rows are being nudged in the opposite direction of the blame.")

            # --- POST-UPDATE CHECK ---
            # Re-run forward to see improvement
            v_i_new = self.emb_weights[idx_i:idx_i+1]
            v_love_new = self.emb_weights[idx_love:idx_love+1]
            combined_new = np.concatenate([v_i_new, v_love_new], axis=1)
            scores_new = np.dot(combined_new, self.dense_weights) + self.dense_biases
            
            print(f"\n✅ AFTER UPDATE: Score for 'ai' moved from {scores[0, 2]:.4f} -> {scores_new[0, 2]:.4f}")
            if np.argmax(scores_new) == 2:
                print("🏆 SUCCESS! 'ai' is now the winner.")

        print("\n" + "="*80)
        print("🧪 THE VECTOR SHIFT: HOW 'LOVE' CHANGED")
        print("="*80)
        # We'll use the initial state vs final state
        # (Assuming we saved the initial state or just showing the final vs a reference)
        print(f"Final 'love' vector: {self.emb_weights[idx_love]}")
        print("Logic: Those numbers have physically moved so that when they hit the Brain,")
        print("       they generate a high score for 'ai'.")
        print("       THIS is how a computer 'learns' the context of a word.")

if __name__ == "__main__":
    xray = VerbalBackpropXRay()
    # Let's save the original love vector for comparison
    original_love = xray.emb_weights[1].copy()
    
    xray.train(epochs=10)
    
    print(f"\n📈 SHIFT SUMMARY:")
    print(f"Original 'love': {original_love}")
    print(f"Final 'love':    {xray.emb_weights[1]}")
    print(f"Difference:      {xray.emb_weights[1] - original_love}")
