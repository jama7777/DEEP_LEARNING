# 🧬 Theory Note 11: Verbal Identity & Semantic Drift (2026-05-08)

This note documents the "Soul" of the Verbal Path: how raw IDs become meaningful concepts in a 128D universe.

---

## 🏗️ 1. The Embedding as a "Malleable Identity"
We learned that an **Embedding** is not a static label. It is a **Weight** that lives in its own layer.

*   **The Look-up Table:** Every word has its own row of coordinates (e.g., 128 dimensions).
*   **The Molding Machine:** The **Dense Layer** acts as a "Teacher." It is shared by all words. 
*   **The Deep Truth:** Similar words (Sun and Moon) become similar because the **Shared Teacher** forces their **Unique Embeddings** into the same shape to satisfy the same goal (e.g., "Predict Brightness").

---

## 🌌 2. Semantic Drift (The Error Magnet)
We visualized how words "travel" through space during training.

1.  **Starting Point:** Random noise. No meaning.
2.  **The Force (Backprop):** Every time a word is seen in a context, the "Incoming Error" ($dout$) acts like a **Magnet**, pulling the word's coordinates toward the "Truth."
3.  **Clustering:** If "Sun" and "Moon" both appear with the word "Rises," they are pulled by the same magnet. Eventually, they become **Mathematical Neighbors**.

---

## ⚖️ 3. The 128D Tug-of-War
How can two words be "Similar" and "Different" at the same time?

*   **High-Dimensional Space:** With 128 dimensions, a word has 128 "rooms" to store facts.
*   **Dimensional Partitioning:**
    *   **Room 0-50:** Both words store "I am Bright." (Similarity = 1.0)
    *   **Room 51-100:** Sun stores "Hot", Moon stores "Cold." (Similarity = -1.0)
*   **The Result:** The final **Cosine Similarity** is the average of all these rooms. They are "Related but Distinct."

---

## 🏁 Summary of the "Verbal" Journey
| Concept | Role |
| :--- | :--- |
| **Embedding** | The "Identity" of the word (Where it lives). |
| **Dense Weights** | The "Thinking" of the model (What it expects). |
| **Backprop ($dout \cdot W^T$)** | The "Translation" that tells the Identity how to change based on the Thinking. |
