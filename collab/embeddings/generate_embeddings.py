import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import gzip


def generate_embeddings(
    drug_info_file: str = "data/ogbl_ddi/ogbl_ddi/mapping/ddi_description.csv.gz",
    output_file: str = "cache/biokg_embeddings.pt",
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cuda"
) -> torch.Tensor:
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    print(f"Loading drug info from: {drug_info_file}")
    with gzip.open(drug_info_file, 'rt') as f:
        df = pd.read_csv(f)
    
    print(f"Generating embeddings for {len(df)} drugs...")
    drug_texts = df['description'].fillna('').tolist()
    
    embeddings = model.encode(drug_texts, convert_to_tensor=True, show_progress_bar=True)
    
    print(f"Saving embeddings to: {output_file}")
    torch.save(embeddings, output_file)
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    generate_embeddings()
