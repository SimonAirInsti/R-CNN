import argparse
import os
import torch
import torch.nn as nn
import esm
import pandas as pd
from Bio import SeqIO

# ==========================================
# 1. ORIGINAL ARCHITECTURE (RCNN.py)
# ==========================================
class RCNN(nn.Module):
    """
    RCNN for sequence-level binary classification from per-residue ESM-2 embeddings.
    """
    def __init__(
        self,
        input_dim: int = 1280,
        num_filters: int = 128,
        kernel_sizes: list = None,
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        # --- Convolutional feature extraction (multi-scale) ---
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, num_filters, ks, padding=ks // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for ks in kernel_sizes
            ]
        )

        conv_out_dim = num_filters * len(kernel_sizes)

        # --- Bidirectional LSTM ---
        self.rnn = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # avg_pool || max_pool  x  bidirectional
        pool_dim = rnn_hidden * 2 * 2

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # --- Conv1D expects [B, C, L] ---
        x_t = x.permute(0, 2, 1)
        conv_outs = [block(x_t) for block in self.conv_blocks]
        x_t = torch.cat(conv_outs, dim=1)          # [B, conv_out_dim, L]
        x_seq = x_t.permute(0, 2, 1)               # [B, L, conv_out_dim]

        # --- Packed BiLSTM ---
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # --- Masked global pooling ---
        lengths_d = lengths.to(rnn_out.device).float()
        mask = (
            torch.arange(rnn_out.size(1), device=rnn_out.device).unsqueeze(0)
            < lengths_d.unsqueeze(1).long()
        ).unsqueeze(2)                              # [B, L, 1]

        avg_pool = (rnn_out * mask.float()).sum(1) / lengths_d.unsqueeze(1)

        neg_inf = torch.full_like(rnn_out, float("-inf"))
        rnn_masked = torch.where(mask.expand_as(rnn_out), rnn_out, neg_inf)
        max_pool = rnn_masked.max(dim=1).values

        pooled = torch.cat([avg_pool, max_pool], dim=1)   # [B, pool_dim]
        return self.classifier(pooled).squeeze(-1)


# ==========================================
# 2. PROCESSING FUNCTIONS
# ==========================================
def parse_input_file(file_path):
    """Reads input file (FASTA or CSV) and returns a list of tuples (id, sequence)."""
    sequences = []
    _, ext = os.path.splitext(file_path.lower())
    
    if ext in ['.fasta', '.fa']:
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append((record.id, str(record.seq)))
    elif ext == '.csv':
        df = pd.read_csv(file_path)
        if 'sequence' not in df.columns:
            raise ValueError("CSV file must contain a 'sequence' column.")
        for index, row in df.iterrows():
            seq_id = row['id'] if 'id' in df.columns else f"seq_{index}"
            sequences.append((seq_id, row['sequence']))
    else:
        raise ValueError("Unsupported file format. Use .fasta, .fa or .csv")
        
    return sequences

def get_esm_sequence_embeddings(sequences, model_esm, batch_converter, device):
    """Extracts ESM embeddings and returns sequence lengths."""
    _, _, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33], return_contacts=False)
    
    token_representations = results["representations"][33]
    
    embeddings = []
    lengths = []
    
    for i, (_, seq) in enumerate(sequences):
        seq_len = len(seq)
        # Extract residue tokens (exclude <cls> and <eos>)
        seq_rep = token_representations[i, 1 : seq_len + 1] # Shape: [seq_len, 1280]
        embeddings.append(seq_rep)
        lengths.append(seq_len)
        
    return embeddings, lengths


# ==========================================
# 3. MAIN FUNCTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Predict sequence antigenicity using RCNN and ESM-2.")
    parser.add_argument("input_file", help="Path to input FASTA or CSV file.")
    parser.add_argument("--model_path", default="best_model.pt", help="Path to the .pt model file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    print(f"Loading sequences from {args.input_file}...")
    sequences = parse_input_file(args.input_file)
    if not sequences:
        print("No sequences found.")
        return

    print("Loading ESM-2 language model...")
    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model_esm = model_esm.to(device)
    model_esm.eval()

    print(f"Loading RCNN model from {args.model_path}...")
    custom_model = RCNN()
    
    # Load state dict with potential 'module.' prefix removal
    state_dict = torch.load(args.model_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    custom_model.load_state_dict(state_dict)
    custom_model = custom_model.to(device)
    custom_model.eval()

    print("Generating embeddings and performing inference...\n")
    print("-" * 50)
    
    # Sequence-by-sequence inference
    for seq_id, seq in sequences:
        
        # 1. Get embedding [seq_len, 1280] and length
        embeddings_list, lengths_list = get_esm_sequence_embeddings([(seq_id, seq)], model_esm, batch_converter, device)
        
        # 2. Format for RCNN: add batch dimension -> [1, seq_len, 1280]
        input_tensor = embeddings_list[0].unsqueeze(0).to(device)
        lengths_tensor = torch.tensor(lengths_list, dtype=torch.long).to(device)
        
        # 3. Prediction
        with torch.no_grad():
            logits = custom_model(input_tensor, lengths_tensor)
            
            probability = torch.sigmoid(logits).item()
            is_antigenic = probability >= 0.5 
            
            result = "Yes" if is_antigenic else "No"
            print(f"ID: {seq_id} | Antigenic: {result} (Score: {probability:.4f})")
            
    print("-" * 50)
    print("Inference completed.")

if __name__ == "__main__":
    main()