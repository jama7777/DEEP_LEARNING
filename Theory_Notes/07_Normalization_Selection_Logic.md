# Normalization Selection: BatchNorm vs LayerNorm

In deep learning, "Normalization" is the act of re-centering and re-scaling data to keep the math stable. The choice of *which* normalization to use depends on how you "slice" the data.

## 🥊 The Selection Battleground

| Feature | BatchNorm (BN) | LayerNorm (LN) |
| :--- | :--- | :--- |
| **Slice Direction** | **Vertical** (Across the Batch) | **Horizontal** (Across Features) |
| **Logic** | "Compare this sample to the rest of the group." | "Normalize this sample using only itself." |
| **Best For** | Computer Vision (CNNs) | NLP (Transformers, RNNs) |
| **Dependency** | Dependent on Batch Size. | **Independent** of Batch Size. |

### 🚨 The "Small Batch" Risk
If your batch size is too small (e.g., 1 or 2), BatchNorm becomes "shaky" because the average of 1 person isn't representative of a population. **LayerNorm** solves this by normalizing based only on the features of that single sample.

### 💡 Selection Rule of Thumb
1. **Working with sequences/text?** -> Use LayerNorm.
2. **Working with images/large batches?** -> Use BatchNorm.
3. **Inference/Real-time?** -> Use LayerNorm (since you often process 1 sample at a time).
