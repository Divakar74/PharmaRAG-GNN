import torch
from torch_geometric.data import Data
from typing import Tuple, Dict
import gzip
import pickle


class DDIDataset:
    """
    Dataset manager for DDI prediction data.
    Handles loading and preprocessing of OGB-DDI dataset.
    """
    
    def __init__(self, data_path: str = "./data/ogbl_ddi"):
        self.data_path = data_path
    
    def load_graph(self) -> Tuple[Data, Dict]:
        """
        Load DDI graph from processed data.
        
        Returns:
            Tuple of (Data object, split edges dict)
        """
        processed_file = f"{self.data_path}/ogbl_ddi/processed/geometric_data_processed.pt"
        data = torch.load(processed_file)
        
        split_file = f"{self.data_path}/ogbl_ddi/split/target"
        split_edge = torch.load(split_file)
        
        return data, split_edge
    
    def load_drug_mapping(self) -> Dict[int, str]:
        """
        Load drug ID to name mapping.
        
        Returns:
            Dict mapping node index to drug ID
        """
        mapping_file = f"{self.data_path}/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
        import pandas as pd
        
        with gzip.open(mapping_file, 'rt') as f:
            df = pd.read_csv(f)
        
        idx_to_drug = {row['node idx']: row['drug id'] for _, row in df.iterrows()}
        return idx_to_drug


def load_ogbl_ddi(data_path: str = "./data/ogbl_ddi") -> Tuple[Data, Dict]:
    """
    Convenience function to load OGBL-DDI dataset.
    
    Args:
        data_path: Path to dataset directory
    
    Returns:
        Tuple of (graph Data, split edges dict)
    """
    dataset = DDIDataset(data_path)
    return dataset.load_graph()


def load_ogbl_biokg(data_path: str = "./data/biokg") -> Data:
    """
    Load OGBL-BioKG dataset.
    
    Args:
        data_path: Path to BioKG data directory
    
    Returns:
        BioKG graph Data object
    """
    # Implementation depends on BioKG data format
    # This is a placeholder
    raise NotImplementedError("BioKG loading not yet implemented")
