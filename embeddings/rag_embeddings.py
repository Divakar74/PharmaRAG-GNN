import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings


class RAGEmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda",
        chroma_path: str = "./chroma_db",
        collection_name: str = "biokg_drugs"
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except:
            self.collection = self.chroma_client.create_collection(collection_name)
    
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings
    
    def embed_drug(self, drug_info: Dict) -> torch.Tensor:
        text = self._format_drug_text(drug_info)
        return self.embed_texts([text])[0]
    
    def _format_drug_text(self, drug_info: Dict) -> str:
        parts = []
        if 'name' in drug_info:
            parts.append(f"Drug: {drug_info['name']}")
        if 'description' in drug_info:
            parts.append(f"Description: {drug_info['description']}")
        if 'mechanism' in drug_info:
            parts.append(f"Mechanism: {drug_info['mechanism']}")
        if 'targets' in drug_info:
            parts.append(f"Targets: {', '.join(drug_info['targets'])}")
        return " ".join(parts)
    
    def add_to_index(self, ids: List[str], documents: List[str], metadata: Optional[List[Dict]] = None):
        self.collection.add(ids=ids, documents=documents, metadatas=metadata)
    
    def search(self, query: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> Dict:
        results = self.collection.query(query_texts=[query], n_results=n_results, where=filter_dict)
        return results
    
    def get_embeddings_count(self) -> int:
        return self.collection.count()
