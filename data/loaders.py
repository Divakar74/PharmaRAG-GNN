"""
Data loading utilities for DDI prediction.
"""
import torch
from typing import Tuple, Dict
import gzip
import pandas as pd


def load_ogbl_ddi(data_path: str = "./data/ogbl_ddi") -> Tuple:
    """
    Load OGBL-DDI dataset.
    
    Args:
        data_path: Path to dataset directory
    
    Returns:
        Tuple of (graph Data, split edges dict)
    """
    processed_file = f"{data_path}/ogbl_ddi/processed/geometric_data_processed.pt"
    data = torch.load(processed_file)
    
    split_file = f"{data_path}/ogbl_ddi/split/target"
    split_edge = torch.load(split_file)
    
    return data, split_edge


def load_ogbl_biokg(data_path: str = "./data/biokg") -> Tuple:
    """
    Load OGBL-BioKG dataset.
    
    Args:
        data_path: Path to BioKG data directory
    
    Returns:
        BioKG graph Data object
    """
    # Implementation depends on BioKG data format
    raise NotImplementedError("BioKG loading not yet implemented")


def load_drug_mapping(data_path: str = "./data/ogbl_ddi") -> Dict[int, str]:
    """
    Load drug ID to name mapping.
    
    Args:
        data_path: Path to dataset directory
    
    Returns:
        Dict mapping node index to drug ID
    """
    mapping_file = f"{data_path}/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
    
    with gzip.open(mapping_file, 'rt') as f:
        df = pd.read_csv(f)
    
    idx_to_drug = {row['node idx']: row['drug id'] for _, row in df.iterrows()}
    return idx_to_drug
