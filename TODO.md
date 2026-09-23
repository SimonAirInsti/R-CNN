## Recommendations (Priority Order)
- ~~Establish baselines: Compare against simpler models (logistic regression, SVM, random forest on mean-pooled ESM embeddings).~~
- Optimize decision threshold: Use ROC curve analysis on validation data to find the optimal operating point.
- Add attention/interpretability: Implement attention pooling to identify antigenic regions.
- Complete CV for all variants: Run 10-fold CV for each architecture variant to make a fair comparison.
- ~~Document dataset source: Cite the provenance of Tables S1-S4 and describe the curation methodology.~~
- Consider multi-layer ESM embeddings: Concatenate representations from multiple layers.
- Add learning rate scheduling.
- ~~Document project with proper README.~~
