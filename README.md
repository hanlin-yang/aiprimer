# Bio-AI-SaaS Primer Design API

Enterprise-grade PCR primer design service powered by AI and bioinformatics algorithms.

## Overview

This is a FastAPI-based RESTful API for automated PCR primer design, combining:

- **AI-Powered Design**: ReAct agent loop using Claude/OpenAI LLMs for intelligent natural language request parsing and primer optimization
- **Bioinformatics Core**: Primer3 for thermodynamic calculations, BLAST+ for specificity checking
- **Variant Integration**: dbSNP/gnomAD integration for SNP conflict detection
- **Regulatory Compliance**: FDA 21 CFR Part 11 compliant audit trail
- **Biosecurity**: Input sanitization and security guardrails

## Features

### Primer Design Capabilities
- **qPCR**: Quantitative PCR primers (70-300 bp amplicons)
- **PCR**: Standard PCR primers
- **LAMP**: Loop-mediated isothermal amplification
- **NGS**: Next-generation sequencing library prep

### Reference Genomes
- Human: hg38, hg19
- Mouse: mm10, mm39

### Analysis & Reports
- Thermodynamic analysis (Tm, ΔG, GC content)
- Specificity verification via BLAST
- SNP/variant conflict detection
- HTML/PDF report generation
- In silico gel visualization

## Project Structure

```
aiprimer-main/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── core/
│   ├── agent.py            # ReAct agent orchestrator
│   ├── guardrails.py       # Security & input validation
│   └── config.py           # Configuration management
├── services/
│   ├── bio_compute.py      # Primer3, BLAST, variant checking
│   └── report_generator.py # HTML/PDF report generation
├── schemas/
│   ├── primer.py           # Pydantic models for primer design
│   └── variant.py          # Variant and SNP schemas
├── prompts/
│   └── versions.yaml       # Versioned LLM prompt templates
├── templates/
│   └── primer_report.html  # Jinja2 report template
├── tests/                  # Test suite
└── utils/                  # Utility functions
```

## Installation

### Prerequisites
- Python 3.10+
- Redis (for async task queue)
- BLAST+ (for specificity checking)
- Primer3 library

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd aiprimer-main

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with the following variables:

```env
# LLM API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# BLAST+ Configuration
BLAST_DB_PATH=/path/to/blast/databases

# Security
SECRET_KEY=your_secret_key
```

## Running the Application

### Development

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root info |
| `/health` | GET | Health check |
| `/api/v1/primers/design` | POST | Design primers (main endpoint) |
| `/api/v1/primers/batch` | POST | Batch primer design (async) |
| `/api/v1/primers/batch/{batch_id}` | GET | Check batch status |
| `/api/v1/reports/generate` | POST | Generate HTML/PDF report |
| `/api/v1/audit/events` | GET | Audit trail (FDA 21 CFR Part 11) |
| `/api/v1/security/events` | GET | Security audit log |
| `/api/v1/utils/validate-sequence` | POST | Validate DNA sequence |
| `/api/v1/utils/databases` | GET | List available reference genomes |

## API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Usage Example

### Design Primers via API

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/primers/design",
    json={
        "sequence": "ATGCGATCGATCGATCG...",
        "task_type": "qPCR",
        "constraints": {
            "tm_min": 58.0,
            "tm_max": 62.0,
            "gc_min": 40.0,
            "gc_max": 60.0,
            "product_size_min": 100,
            "product_size_max": 200
        },
        "reference_genome": "hg38",
        "check_variants": True
    }
)

result = response.json()
print(result["primer_pairs"])
```

### Natural Language Request

```python
response = requests.post(
    "http://localhost:8000/api/v1/primers/design",
    json={
        "natural_language_request": "Design qPCR primers for BRCA1 exon 11 with Tm around 60°C",
        "reference_genome": "hg38"
    }
)
```

## Core Components

### ReAct Agent (`core/agent.py`)
Implements a Reasoning + Acting loop for intelligent primer design:
- Tool-calling architecture with Primer3, BLAST, and Variant Checker
- Natural language understanding for request parsing
- Automatic retry with feedback-driven optimization

### Security Guardrails (`core/guardrails.py`)
- Input validation against prompt injection attacks
- Biosecurity screening for dangerous sequences
- Rate limiting
- Audit logging for regulatory compliance

### Bio Compute Services (`services/bio_compute.py`)
- **Primer3Wrapper**: Thermodynamic calculations using Nearest-Neighbor model
- **LocalBlastService**: Specificity analysis and off-target detection
- **VariantChecker**: SNP/variant conflict detection with dbSNP/gnomAD

### Report Generator (`services/report_generator.py`)
- HTML and PDF report generation
- Comprehensive design documentation
- Audit trail inclusion

## Dependencies

### Core
- FastAPI, Uvicorn, Pydantic - Web framework
- primer3-py, Biopython - Bioinformatics
- Anthropic SDK, OpenAI SDK - LLM integration

### Infrastructure
- Celery, Redis - Async task processing
- Jinja2, WeasyPrint - Report generation
- structlog, prometheus-client - Monitoring

### Security
- python-jose, passlib - Authentication

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
