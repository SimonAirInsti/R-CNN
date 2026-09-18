## Recommendations (Priority Order)
~~1. Establish baselines: Compare against simpler models (logistic regression, SVM, random forest on mean-pooled ESM embeddings).~~
2. Optimize decision threshold: Use ROC curve analysis on validation data to find the optimal operating point.
3. Add attention/interpretability: Implement attention pooling to identify antigenic regions.
4. Complete CV for all variants: Run 10-fold CV for each architecture variant to make a fair comparison.
~~5. Document dataset source: Cite the provenance of Tables S1-S4 and describe the curation methodology.~~
6. Consider multi-layer ESM embeddings: Concatenate representations from multiple layers.
7. Add learning rate scheduling.
~~8. Document project with proper README.~~