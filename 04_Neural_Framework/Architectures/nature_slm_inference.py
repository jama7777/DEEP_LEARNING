import numpy as np
import os
from nature_slm_master import Nature_SLM_Pro, get_data

def generate_text(model, w2i, i2w, prompt, max_gen=20, context_size=5, temperature=1.0):
    """
    Performs inference to generate text based on a prompt.
    """
    # 1. Tokenize prompt
    words = prompt.lower().split()
    input_ids = [w2i[w] for w in words if w in w2i]
    
    if not input_ids:
        # Fallback if no words in vocab
        input_ids = [np.random.randint(0, len(w2i))]
    
    # Pad if prompt is shorter than context_size
    if len(input_ids) < context_size:
        input_ids = [0] * (context_size - len(input_ids)) + input_ids
    
    generated_words = list(words)
    
    print(f"\n🔮 Generating from: '{prompt}'...")
    
    for _ in range(max_gen):
        # 2. Get the current context window (last context_size words)
        window = np.array([input_ids[-context_size:]])
        
        # 3. Forward Pass
        probs = model.forward(window)[0]
        
        # 4. Sampling (Temperature)
        if temperature <= 0:
            next_id = np.argmax(probs)
        else:
            # Apply temperature scaling
            probs = np.log(probs + 1e-15) / temperature
            exp_probs = np.exp(probs - np.max(probs))
            probs = exp_probs / np.sum(exp_probs)
            next_id = np.random.choice(len(probs), p=probs)
        
        # 5. Update state
        input_ids.append(next_id)
        generated_words.append(i2w[next_id])
        
        # Break if we hit a period (optional)
        # if i2w[next_id] == ".": break
        
    return " ".join(generated_words)

def run_inference():
    # 1. Setup paths
    base_dir = "/Users/indra/Desktop/DEEP_LEARNING/04_Neural_Framework/Architectures"
    checkpoint_path = os.path.join(base_dir, "nature_slm_checkpoint.npz")
    
    # 2. Load Vocab and Mappings
    print("📖 Loading Vocabulary...")
    data_ids, vocab, w2i, i2w = get_data()
    vocab_size = len(vocab)
    context_size = 5
    
    # 3. Initialize Model & Load Weights
    print("🧠 Initializing Nature SLM...")
    model = Nature_SLM_Pro(vocab_size, emb_dim=128, context_size=context_size, hidden_dim=256)
    
    if os.path.exists(checkpoint_path):
        model.load(checkpoint_path)
    else:
        print("⚠️ Warning: No checkpoint found! Model will generate garbage.")
    
    # 4. Interactive Generation
    print("\n✨ Nature SLM Inference Engine Ready")
    print("Type 'exit' to quit.")
    print("-" * 40)
    
    while True:
        prompt = input("\nEnter prompt (at least 5 words recommended): ")
        if prompt.lower() == 'exit':
            break
            
        output = generate_text(
            model, w2i, i2w, prompt, 
            max_gen=30, 
            context_size=context_size,
            temperature=0.7 # lower = more focused, higher = more creative
        )
        
        print(f"\nResult: \033[92m{output}\033[0m")
        print("-" * 40)

if __name__ == "__main__":
    run_inference()
