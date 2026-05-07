# 📓 Verbal Foundations: Words as Vectors
**Date:** May 7, 2026
**Topic:** How Language becomes Math (Embeddings)

---

## 🏗️ 1. The Vocabulary (The World Map)
Before a network can "speak," it needs a list of all words it knows. This is the **Vocabulary**.
*   `0: <PAD>` (Empty space)
*   `1: I`
*   `2: Love`
*   `3: Deep`
*   `4: Learning`

## 👑 2. The King-Queen Algebra
This is the most famous part of "Verbal" math. In an embedding space, words with similar meanings sit close together.
*   **Equation**: `King - Man + Woman ≈ Queen`
*   **Deep Point**: The network doesn't know what a "King" is, but it knows that the **Relationship** between King/Queen is the same as the relationship between Man/Woman.

## 📍 3. The Embedding Layer (The Lookup Table)
An Embedding layer is a special type of **Dense Layer** that act as a dictionary.
*   **Input**: A word index (like `4`).
*   **Output**: A list of weights (like `[0.5, -0.2, 0.1]`).
*   **The "Push"**: During training, if the network gets a sentence wrong, it "pushes" the word coordinates closer to the words that would have made the sentence correct.

## 📉 4. One-Hot vs. Dense Embeddings
*   **One-Hot**: `[0, 0, 0, 1, 0]` -> Hard to compute, no relationships.
*   **Dense**: `[0.55, -0.12]` -> Efficient, stores "shades" of meaning.

---

## 🕵️ 5. The Chain of Blame (Backpropagation)
When the model predicts "Love" instead of "AI," it calculates the **Gradient** (The Blame).
*   **The Detective**: Backprop is a detective investigating a crime. It starts at the final output and travels backward along the "wires" (weights) to find who to blame.
*   **Matrix Math**: 
    *   `dL/dW`: Fixing the **Brain** (How to turn the knobs).
    *   `dL/dX`: Fixing the **Message** (Sending a warning to the dictionary).
*   **The Slicer**: Since words are glued together (Flattened) for the forward pass, we must **slice** the blame in half to tell each word specifically how it failed.

## 🤝 6. The Batch Consensus (Parallel Learning)
Instead of training one sentence at a time, we use a **Batch**.
*   **The Classroom**: 30 students (The Batch) listen to one teacher (The Weights).
*   **No Information Loss**: Every sentence gets its own unique prediction.
*   **The Compromise**: During the backward pass, we **average** the complaints. This filters out the noise and finds the "General Truth" that works for everyone.

## ✂️ 7. The ReLU Fold (Non-Linearity)
Without ReLU, a deep model is just a flat sheet of paper.
*   **The Origami**: ReLU is the "Fold." It sets negative signals to zero, allowing the logic to "bend."
*   **Decision Power**: It enables **"If/Then" logic**. If a word is irrelevant, ReLU kills the signal.
*   **Survival**: ReLU keeps gradients at a constant strength (1.0), ensuring the "Blame" can travel through 100 layers without fading away.

## 🏎️ 8. The Optimizer (The Smart Driver)
The Optimizer decides how fast to turn the knobs based on the blame.
*   **SGD (Stochastic Gradient Descent)**: A "Skateboard." Simple and fast, but it gets stuck in cracks or bounces around on bumpy roads.
*   **Adam (Adaptive Moment Estimation)**: A "Smart Car." 
    *   **Momentum**: It remembers the past and speeds up in smooth valleys.
    *   **Scaling**: It slows down if the weights are jumping too much (vibrating).
*   **The Goal**: Keeping 1 Billion parameters in sync so they don't explode or vanish.

---
**Summary**: Language Modeling is just **Geometry**. To understand a sentence, the network maps each word into a high-dimensional space and looks for patterns in how those points move together. We use **ReLU** to bend the space, **Backprop** to find the errors, and **Adam** to drive the weights toward the truth.
