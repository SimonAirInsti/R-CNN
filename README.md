# R-CNN for Protein Antigenicity Prediction

## 1. Introduction

Antigenicity is defined as the ability of a macromolecule, typically of foreign origin, to interact with components of the immune system, such as antibodies or major histocompatibility complex (MHC/HLA) receptors. This property is closely related to immunogenicity, understood as the capacity of a substance to induce an effective and lasting immune response in the host, including the generation of memory cells. When a protective immunogen originates from a pathogen, it becomes a potential candidate for vaccine development; thus, the early identification of such protective antigens is a critical step in in silico vaccine design.

Historically, the detection of protective antigens has relied on experimental biological assays, which are notably costly and labor-intensive. This has driven the development of bioinformatics tools for antigenicity prediction, particularly for viral and bacterial pathogen proteins. Among the most widely used are VaxiJen 2.0, Vaxign/Vaxign2 and ANTIGENpro, which have been integrated into numerous multi-epitope vaccine design pipelines. In a comparative analysis of reverse vaccinology tools, VaxiJen 2.0 was identified as one of the methodologies offering the best balance between selected proteome fraction and ability to recover known protective antigens, explaining its widespread adoption. Building on this foundation, VaxiJen 3.0 has achieved substantial improvements in sensitivity and recall over previous versions.

This tool is characterized as an "alignment-independent" predictor that avoids the need for multiple sequence alignments, relying instead on an auto-cross-covariance (ACC) transformation of sequences in terms of physicochemical descriptors. Specifically, amino acid sequences are projected into a space of z-descriptors derived via multidimensional scaling from hundreds of physicochemical properties, originally compiled by Venkatarajan et al. and subsequently expanded in the AAindex database. The preprocessing of protein sequences constitutes a notable challenge due to the large variability in protein length and composition, which hinders the direct application of many classical machine learning algorithms.

However, descriptor-based approaches share a fundamental limitation: they encode amino acid properties independently of sequential context and the evolutionary history of the protein. In other words, the same residue receives the same representation regardless of its surrounding environment, making it difficult to capture long-range interactions, conformational effects, and coevolutionary patterns that are determinants of antigenicity and epitope presentation. This landscape has paved the way for the adoption of protein language models, which bring to biology the advances of large language models.

In this context, ESM-2 (Evolutionary Scale Modeling 2) represents an important milestone in protein processing. ESM-2 is a family of Transformer models trained in a self-supervised manner on millions of protein sequences at evolutionary scale, enabling it to predict masked residues with remarkable reliability and model sequence distributions without recourse to multiple sequence alignments or explicit structural information. As a result, ESM-2 generates dense per-residue embeddings (up to 1280 dimensions in the largest variants), in which information about the intrinsic properties of the amino acid, local sequential context, coevolutionary patterns, and latent structural signals is jointly encoded. These embeddings have proven highly transferable to downstream tasks, including direct 3D structure prediction (via ESMFold), functional protein design through diffusion models, and identification of conserved motifs in intrinsically disordered regions.

The adoption of ESM-2 in immunoinformatics offers several advantages over static physicochemical descriptors: it naturally handles variable-length sequences, integrates global and local information within a single representation, and implicitly captures structural and evolutionary aspects relevant to immune recognition. Consequently, deep learning architectures operating on ESM-2 embeddings, such as Recurrent Convolutional Neural Networks (RCNNs), emerge as a particularly promising alternative to overcome the limitations of classical descriptor-based models within reverse vaccinology pipelines and rational vaccine design.

Accordingly, the objective of this work was to design a model that applies these emerging technologies to improve antigenicity prediction, rigorously evaluate the model's performance, and integrate its functionality within a bioinformatics pipeline developed for this purpose. The ESM-2 protein language model was employed in the sequence processing stage, standardized artificial intelligence model evaluation techniques — such as stratified cross-validation — were applied, and Docker and Nextflow were used to ensure reproducibility and portability of the analyses. The methodology, results, and conclusions are presented in the following sections of this document.

## 2. Sequence Representation via ESM-2

The first step of the workflow consists of transforming the primary amino acid sequence, typically stored in FASTA format, into a rich mathematical representation exploitable by deep learning models. For this purpose, the ESM-2 protein language model is used, which treats the sequence as a "sentence" whose vocabulary is formed by the twenty canonical amino acids. Applying ESM-2 to the viral sequences in the dataset generates per-residue representations (embeddings) rather than physicochemical scores. In the variant employed (esm2_t33_650M_UR50D), each amino acid is projected into a dense 1280-dimensional vector, whose values depend not only on the residue identity but also on its sequential context and coevolutionary patterns captured during pretraining. This high dimensionality allows implicit encoding of information about the protein's potential structural and functional interactions, offering a far more expressive representation than the static physicochemical descriptors used in classical antigenicity models.

From a computational standpoint, embedding generation via an ESM-2 model constitutes the most demanding stage of the workflow. To make it viable, GPU acceleration is employed, delegating the computation of massive matrix operations to an NVIDIA GeForce RTX 4090 with 24 GB of video memory, optimized for high-performance computing tasks. The combined use of PyTorch and CUDA exploits this infrastructure transparently, drastically reducing processing times compared to a purely CPU-based execution.

Once computed, embeddings are not recalculated at every model run. Instead, they are serialized and stored as optimized tensors. This caching strategy allows the training script to immediately load each protein's vectors into memory, ensuring a smooth transition between the feature extraction phase and the deep learning phase.

Embedding generation is handled by `scripts/generate_embeddings.py`, which iterates over all FASTA files in `data/fasta/`, extracts per-residue representations from layer 33 of ESM-2, and saves them as `.pt` files in `data/embeddings/`.

## 3. Model Architecture

Since ESM-2 produces per-residue embeddings that form a sequence of high-dimensional vectors, the input to the classifier is a variable-length sequence in which both local motifs and long-range dependencies carry biologically relevant signal. The chosen architecture combines 1D convolutional layers with a bidirectional LSTM, each component addressing a complementary aspect of the problem. The convolutional layer extracts local patterns at multiple scales through parallel kernels of different widths (3, 5, and 7 residues). This is biologically motivated: antigenic motifs and physicochemical signatures operate at varying spatial ranges — from short physicochemical propensities (hydrophilicity, charge clusters) that span a few residues, to longer epitope-like regions that extend across tens of positions. Multi-scale convolutions capture these patterns with translation invariance along the sequence, akin to how the immune system recognizes local sequence features regardless of their absolute position in the protein.

Following convolutional feature extraction, the concatenated multi-scale representations are fed into a bidirectional LSTM. The recurrent component models sequential dependencies beyond the limited receptive field of the convolutions, capturing how residues at distant positions co-determine local conformational and functional properties. The bidirectional design ensures that both upstream and downstream context inform the representation of each position. This is essential for antigenicity, where the structural and evolutionary context of a residue often depends on interactions that span the entire protein chain.

The sequence of LSTM hidden states is then reduced to a fixed-length representation through masked global pooling. Both average pooling and max pooling are computed over the actual (non-padded) sequence positions, then concatenated. Average pooling captures the global protein-level profile — properties that are distributed across the entire sequence — while max pooling isolates the most salient local activations, effectively highlighting regions with the strongest antigenic signal. The concatenation of both pooling strategies provides the classifier with complementary information: the overall biophysical landscape of the protein and the presence of distinctive antigenic features, regardless of their position within the sequence.

The pooled representation is passed through a two-layer fully connected classifier head with ReLU activation and dropout regularization, producing a single logit that, after sigmoid transformation, yields the probability that the protein is antigenic.

```
Input: [batch, seq_len, 1280]     -- ESM-2 per-residue embeddings
  │
  ├─ Multi-scale 1D Conv (k=3,5,7) ──→ BatchNorm → ReLU → Dropout
  │     Output: [batch, 3×filters, seq_len]
  │
  ├─ Bidirectional LSTM (2 layers)
  │     Output: [batch, seq_len, 2×hidden]
  │
  ├─ Masked Global Pooling
  │     avg_pool ⊕ max_pool → [batch, 4×hidden]
  │
  ├─ Classifier Head
  │     Linear → ReLU → Dropout → Linear → sigmoid
  │     Output: [batch]  probability
```

### 3.1 Architectural Variants

Five architectural configurations are defined in `config/rcnn_config.yaml` to explore the effect of depth, width, and receptive field on predictive performance:

| Variant | Conv. Blocks | LSTM Layers | Filters | Hidden | Dropout | Parameters |
|---|---|---|---|---|---|---|
| `base` | 3 (k=3,5,7) | 2 | 128 | 128 | 0.30 | 3.4 M |
| `deep_conv_lstm` | 4 (k=3,5,7,9) | 3 | 192 | 192 | 0.35 | 9.3 M |
| `wide_bilstm` | 3 (k=3,5,7) | 3 | 160 | 160 | 0.25 | 5.2 M |
| `extra_variant_1` | 2 (k=3,5) | 2 | 96 | 96 | 0.20 | 1.5 M |
| `extra_variant_2` | 5 (k=3,5,7,9,11) | 4 | 224 | 224 | 0.40 | 16.3 M |

The base model provides the reference architecture. The variants systematically vary the number of convolutional blocks and kernel sizes (receptive field), LSTM depth, channel width, and dropout rate, to assess whether additional capacity translates to improved generalization.

## 4. Project Structure

```
R-CNN/
├── config/
│   └── rcnn_config.yaml          # Centralized hyperparameters and paths
├── data/
│   ├── fasta/                    # Raw FASTA files (Tables S1–S4)
│   └── embeddings/               # Precomputed ESM-2 .pt files
├── scripts/
│   ├── generate_embeddings.py    # ESM-2 embedding extraction
│   ├── train.py                  # Architecture comparison entry point
│   └── cross_validate.py         # Stratified k-fold cross-validation
├── src/modules/
│   ├── rcnn_model/               # RCNN architecture definition
│   ├── rcnn_training/            # Training, evaluation, CV routines
│   └── rcnn_inference/           # Single-sequence prediction
├── shared/
│   └── config_loader.py          # YAML configuration utilities
├── legacy/                       # Pre-modularization code (preserved)
├── results/                      # Training artifacts and reports
└── requirements.txt
```

## 5. Configuration

All hyperparameters are centralized in `config/rcnn_config.yaml`. The file defines:

- **`project`**: output directories and data paths.
- **`embedding`**: ESM-2 model name, device, and I/O paths.
- **`model.base`**: reference architecture hyperparameters.
- **`model.candidate_architectures`**: named variant configurations.
- **`training`**: epochs, batch size, learning rate, weight decay, early stopping patience, class weights.
- **`validation`**: decision threshold and random seed.
- **`paths`**: file paths for training and test embeddings (Tables S1–S4).

## 6. Usage

### 6.1 Generate ESM-2 embeddings

```bash
python scripts/generate_embeddings.py
```

Processes all FASTA files in `data/fasta/` and saves per-residue embeddings as `.pt` files in `data/embeddings/`.

### 6.2 Compare architectures

```bash
python scripts/train.py
```

Trains all variants declared in `config/rcnn_config.yaml` on a single 80/20 train/validation split, evaluates on the held-out test set, and produces a comparison report (JSON, CSV, markdown table, bar chart) under `results/RCNN/`.

### 6.3 Run cross-validation

```bash
python scripts/cross_validate.py --model-path results/RCNN/base_best_model.pt
```

Performs 10-fold stratified cross-validation for the architecture identified by the checkpoint. Outputs per-fold metrics, aggregate statistics with SEM, a loss-per-epoch plot with 99% confidence intervals, and the best-fold checkpoint under `results/RCNN/`.

### 6.4 Inference

The inference module (`src/modules/rcnn_inference/entrypoint.py`) provides functions to load a saved model and predict the antigenicity probability for a single sequence given its ESM-2 embedding tensor.

## 7. Evaluation Metrics

The following metrics are reported for each evaluation:

- **Accuracy**: overall proportion of correct predictions.
- **Precision**: proportion of predicted antigens that are truly antigenic.
- **Recall (Sensitivity)**: proportion of true antigens that are correctly identified.
- **Specificity**: proportion of true non-antigens correctly identified.
- **Matthews Correlation Coefficient (MCC)**: correlation between predicted and observed labels; robust to class imbalance.
- **AUC-ROC**: area under the receiver operating characteristic curve; threshold-independent ranking performance.

Cross-validation results report mean ± standard error of the mean (SEM) across folds.

## 8. Dataset

The protein sequences are sourced from the supplementary tables of the reference publication (Tables S1–S4), comprising:

- **Table S1**: Training set of viral antigens.
- **Table S2**: Training set of viral non-antigens.
- **Table S3**: Test set of viral antigens.
- **Table S4**: Test set of viral non-antigens.

The train/test split is predefined in the source data and is not altered by this pipeline. The source of set data is:
```cite
Doneva, N., & Dimitrov, I. (2024). Viral immunogenicity prediction by machine learning methods. International journal of molecular sciences, 25(5), 2949.
```

## 9. Reproducibility

- Random seed: 42 (configurable in `rcnn_config.yaml`).
- Embeddings are cached; re-running training does not recompute ESM-2 features.
- All model checkpoints, metric logs, and plots are saved under `results/RCNN/`.
- Training uses PyTorch with AdamW optimizer, gradient clipping (max_norm=1.0), and weighted BCE loss to handle class imbalance.
