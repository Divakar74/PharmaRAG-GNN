import torch
from torch_geometric.data import Data
from typing import Tuple, Dict
import gzip
import pandas as pd


def load_ogbl_ddi(data_path: str = "./data/ogbl_ddi") -> Tuple[Data, Dict]:
    """Load OGBL-DDI dataset."""
    processed_file = f"{data_path}/ogbl_ddi/processed/geometric_data_processed.pt"
    data = torch.load(processed_file)
    split_file = f"{data_path}/ogbl_ddi/split/target"
    split_edge = torch.load(split_file)
    return data, split_edge


def load_drug_mapping(data_path: str = "./data/ogbl_ddi") -> Dict[int, str]:
    """Load drug ID to name mapping."""
    mapping_file = f"{data_path}/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
    with gzip.open(mapping_file, 'rt') as f:
        df = pd.read_csv(f)
    return {row['node idx']: row['drug id'] for _, row in df.iterrows()}
