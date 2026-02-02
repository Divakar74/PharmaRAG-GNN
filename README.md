# Advanced DDI Prediction System

A production-ready Drug-Drug Interaction (DDI) prediction system using LLM-GNN hybrid models with retrieval-augmented generation (RAG), blockchain auditability, and explainable AI.

## Features

- **RAG-Enhanced Embeddings**: Context-aware drug representations from BioKG retrieval using ChromaDB vector search
- **Dual-Graph GNN**: Fuses DDI and BioKG graphs for multi-modal knowledge integration
- **Focal Loss**: Addresses class imbalance in rare DDIs (configurable γ=2.0, α=0.25)
- **Blockchain Auditability**: Immutable logs for predictions and model versions
- **Explainable AI**: Tiered explanation system with LLM-generated insights
- **Production-Ready**: Docker containerization, health checks, error handling

## Quick Start

### Prerequisites

- Python 3.9+
- Neo4j Community Edition (for graph database)
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure your settings in `.env`:
- Neo4j connection details
- OpenAI API key (for LLM explanations)
- Other configuration options

### Running the Application

```bash
# Using the startup script
./scripts/run_app.sh

# Or directly
cd api && python flask_app.py
```

Access the application at `http://localhost:5000`.

## Project Structure

```
app/
├── api/                    # Flask API
│   ├── flask_app.py       # Main Flask application
│   └── frontend/          # React frontend build
├── blockchain/            # Blockchain audit logging
│   └── audit_logger.py    # Audit trail implementation
├── collab/                # Training pipeline
│   ├── configs/           # Experiment configurations
│   ├── data/              # Data loading utilities
│   ├── embeddings/        # LLM embeddings
│   ├── losses/            # Loss functions
│   ├── models/            # GNN models
│   └── train.py           # Training script
├── configs/               # Application configs
├── data/                  # Dataset utilities
├── embeddings/            # RAG embeddings
├── losses/                # Loss functions
├── models/                # Model definitions
├── scripts/               # Shell scripts
├── tests/                 # Unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## API Documentation

### Health Check
```http
GET /api/health
```

### Drug Search
```http
GET /search_drug?q={query}&limit={limit}
```

### DDI Prediction
```http
POST /predict_ddi
Content-Type: application/json

{
  "drug_a": "Aspirin",
  "drug_b": "Warfarin",
  "user_id": "optional_user_id"
}
```

### Generate Explanation
```http
POST /explain
Content-Type: application/json

{
  "drug_a": "Aspirin",
  "drug_b": "Warfarin",
  "probability": 0.85,
  "use_llm": true
}
```

## Model Training

Train the DDI prediction model:
```bash
python collab/train.py
```

Run evaluation:
```bash
python eval.py
```

Generate embeddings:
```bash
python collab/embeddings/generate_embeddings.py
```

## Docker Deployment

```bash
# Build the image
docker build -t ddi-prediction .

# Run the container
docker run -p 5000:5000 ddi-prediction
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Configuration

Edit `configs/models.yaml` to configure:
- Model architecture parameters
- Training hyperparameters
- Database connections
- Embedding settings

## Troubleshooting

### Database Connection Errors
- Ensure Neo4j is running on port 7687
- Verify credentials in connection string
- Check firewall settings

### Model Loading Errors
- Verify model files exist in expected paths
- Check PyTorch version compatibility
- Ensure sufficient RAM for model loading

### API Timeouts
- Increase timeout in client requests
- Check server logs for performance bottlenecks

## License

MIT License

## References

- Base paper: LLM-DDI (Li et al.)
- Datasets: OGB-DDI, OGB-BioKG
- Focal Loss: Lin et al., 2017
- Graph Neural Networks: Kipf & Welling, 2017
