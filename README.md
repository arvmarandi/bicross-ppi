
# BiCrossPPI

Protein–protein interactions (PPIs) are crucial for a wide range of biological processes. While experimental methods for detecting PPIs are costly and time- consuming, recent advances in deep learning and protein language models (PLMs) enable effective PPI prediction from protein sequences. In this work, we propose BiCrossPPI, a bidirectional cross-attention transformer framework for PPI prediction that models the symmetry of protein interactions explicitly. BiCrossPPI first leverages ESMC to obtain multi-level embeddings, followed by a bidirectional cross-attention transformer encoder to capture symmetric mutual dependencies between protein pairs. Then, an attention pooling module transforms encoder
outputs into protein-level embeddings, followed by a lightweight classifier for PPI prediction. Extensive experiments demonstrate that BiCrossPPI consistently outperforms all State of the Art baselines while maintaining a good balance between training efficiency and prediction performance.

# Performance
[Performance against SotA baselines](/figures/BaselinePerformance.jpeg)
