"""Generate ESM-2 residue embeddings from FASTA files and save them in the
format expected by the RCNN training pipeline.

The output is a list of tensors, one per protein, where each tensor has shape
[seq_len, embedding_dim]. The saved .pt file can be loaded by
`load_embeddings()` in the training module.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml
from Bio import SeqIO

try:
    import esm
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "ESM is required. Install the project dependencies in the same Python environment."
    ) from exc


def load_config(config_path: str | os.PathLike[str]) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_device(device_name: str | None) -> torch.device:
    if device_name and device_name.lower() == "cpu":
        return torch.device("cpu")
    if device_name and device_name.lower() == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_residue_embeddings(sequence: str, model_esm, alphabet, device: torch.device) -> torch.Tensor:
    """Compute per-residue ESM-2 embeddings for a single protein sequence."""
    seq = str(sequence).strip()
    if not seq:
        raise ValueError("Empty sequence received.")

    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("seq", seq)])
    batch_tokens = batch_tokens.to(device)

    model_esm.to(device)
    model_esm.eval()

    with torch.no_grad():
        result = model_esm(batch_tokens, repr_layers=[33], return_contacts=False)

    representation = result["representations"][33]
    residue_embeddings = representation[0, 1 : len(seq) + 1, :].cpu()

    if residue_embeddings.shape[0] != len(seq):
        raise RuntimeError(
            f"Embedding length mismatch: expected {len(seq)}, got {residue_embeddings.shape[0]}."
        )

    return residue_embeddings


def save_embedding_file(output_path: str | os.PathLike[str], embeddings: torch.Tensor) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, str(output_path))


def process_fasta_file(fasta_path: str | os.PathLike[str], output_dir: str | os.PathLike[str], model_esm, alphabet, device: torch.device) -> list[torch.Tensor]:
    """Process a FASTA file and save a single aggregated .pt file.

    The output is a Python list of tensors, one tensor per sequence, matching the
    format expected by `load_embeddings()` in the RCNN training pipeline.
    """
    fasta_path = Path(fasta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_embeddings: list[torch.Tensor] = []

    for record in SeqIO.parse(str(fasta_path), "fasta"):
        sequence = str(record.seq).strip()
        if not sequence:
            continue

        embeddings = extract_residue_embeddings(sequence, model_esm, alphabet, device)
        protein_embeddings.append(embeddings)
        print(f"Processed {record.id} | seq_len={len(sequence)} | dim={embeddings.shape[-1]}")

    output_file = output_dir / f"{fasta_path.name}.pt"
    save_embedding_file(output_file, protein_embeddings)
    print(f"Saved aggregated dataset: {output_file} | proteins={len(protein_embeddings)}")
    return protein_embeddings


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "config" / "rcnn_config.yaml")

    embedding_cfg = config.get("embedding", {})
    fasta_dir = project_root / embedding_cfg.get("fasta_dir", "data/fasta")
    output_dir = project_root / embedding_cfg.get("output_dir", "data/embeddings")
    device = get_device(embedding_cfg.get("device"))

    print(f"Using device: {device}")
    print(f"Loading ESM-2 model: {embedding_cfg.get('model_name', 'esm2_t33_650M_UR50D')}")

    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model_esm = model_esm.to(device)
    model_esm.eval()

    for fasta_file in sorted(Path(fasta_dir).glob("*.fasta")):
        print(f"Processing {fasta_file.name}")
        process_fasta_file(
            fasta_path=fasta_file,
            output_dir=output_dir,
            model_esm=model_esm,
            alphabet=alphabet,
            device=device,
        )

    print(f"Embedding generation completed. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
