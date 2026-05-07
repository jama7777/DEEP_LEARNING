# Normalization Selection: BatchNorm vs LayerNorm (2026-05-07)

## 🥊 The Selection Battleground

| Feature | BatchNorm (BN) | LayerNorm (LN) |
| :--- | :--- | :--- |
| **Slice Direction** | **Vertical** (Across the Batch) | **Horizontal** (Across Features) |
| **Logic** | "Compare this sample to the rest of the group." | "Normalize this sample using only itself." |
| **Best For** | Computer Vision (CNNs) | NLP (Transformers, RNNs) |
| **Dependency** | Dependent on Batch Size. | **Independent** of Batch Size. |

### 💡 Selection Rule of Thumb
1. **Working with sequences/text?** -> Use LayerNorm.
2. **Working with images/large batches?** -> Use BatchNorm.
3. **Inference/Real-time?** -> Use LayerNorm.
