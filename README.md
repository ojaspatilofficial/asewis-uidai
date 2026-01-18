# ASEWIS - Aadhar System Engineering & Workflow Intelligence System

A data-intensive hackathon system for UIDAI Aadhar data processing, analytics, and intelligence.

## Project Structure

```
asewis/
│
├── dataset/                          # Data storage
│   ├── raw/                         # Raw, immutable data dump
│   ├── processed/                   # Cleaned and transformed data
│   └── external/                    # External reference data
│
├── data_cache/                       # Temporary data storage
│   ├── embeddings/                  # Vector embeddings cache
│   ├── models/                      # Trained model artifacts
│   └── temp/                        # Temporary processing files
│
├── src/                              # Source code
│   │
│   ├── domain/                      # Domain/Business logic layer
│   │   ├── entities/                # Core business entities
│   │   ├── repositories/            # Repository interfaces
│   │   └── services/                # Domain services
│   │
│   ├── application/                 # Application/Use case layer
│   │   ├── use_cases/               # Application business rules
│   │   ├── dto/                     # Data Transfer Objects
│   │   └── interfaces/              # Application interfaces
│   │
│   ├── infrastructure/              # Infrastructure layer
│   │   ├── database/                # Database connections & models
│   │   ├── repositories/            # Repository implementations
│   │   ├── external_services/       # Third-party integrations
│   │   └── config/                  # Configuration management
│   │
│   ├── data_processing/             # Data processing modules
│   │   ├── pipelines/               # ETL pipelines
│   │   ├── transformers/            # Data transformations
│   │   ├── validators/              # Data validation rules
│   │   └── loaders/                 # Data loading utilities
│   │
│   ├── intelligence/                # ML & Analytics layer
│   │   ├── ml_models/               # Machine learning models
│   │   ├── analytics/               # Statistical analysis
│   │   ├── feature_engineering/     # Feature extraction & creation
│   │   └── predictions/             # Prediction services
│   │
│   └── common/                      # Shared utilities
│       ├── utils/                   # Helper functions
│       ├── constants/               # System constants
│       ├── exceptions/              # Custom exceptions
│       └── logging/                 # Logging configuration
│
├── app/                              # Application interface layer
│   ├── api/                         # REST API
│   │   ├── v1/                      # API version 1
│   │   │   ├── endpoints/           # API endpoints
│   │   │   └── schemas/             # Request/Response schemas
│   │   ├── middleware/              # API middleware
│   │   └── dependencies/            # Dependency injection
│   │
│   └── ui/                          # User Interface
│       ├── static/                  # Static assets
│       ├── templates/               # HTML templates
│       └── components/              # UI components
│
├── notebooks/                        # Jupyter notebooks
│   ├── exploratory/                 # EDA notebooks
│   ├── experiments/                 # ML experiments
│   └── reports/                     # Analysis reports
│
├── tests/                            # Test suite
│   ├── unit/                        # Unit tests
│   ├── integration/                 # Integration tests
│   ├── e2e/                         # End-to-end tests
│   └── fixtures/                    # Test fixtures & mocks
│
├── scripts/                          # Utility scripts
├── docs/                             # Documentation
├── config/                           # Configuration files
└── logs/                             # Application logs
```

## Architecture Principles

- **Clean Architecture**: Separation of concerns with clear boundaries
- **Domain-Driven Design**: Business logic in the domain layer
- **Dependency Inversion**: Dependencies point inward
- **Single Responsibility**: Each module has one reason to change
- **Data Processing Separation**: Independent data pipelines
- **Intelligence Layer**: Isolated ML/Analytics components

## Key Features

- **NASRI Dashboard** - National Aadhaar Service Readiness Index (0-100 score) visualization
- **ASRS Analytics** - Aadhaar Service Risk Score (0-1) for identifying at-risk districts
- **Interactive Choropleth Maps** - District-level visualization across 594 Indian districts
- **AI-Powered Recommendations** - Actionable insights for improving district performance
- **Forecasting Engine** - Predict future trends using statistical models
- **Anomaly Detection** - Flag unusual patterns in service delivery

## Quick Start

### Prerequisites
- Python 3.9+ (3.10 or 3.11 recommended)
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/ojaspatilofficial/asewis-uidai.git
cd asewis-uidai

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Visualization | Plotly, Matplotlib |
| Data Processing | Pandas, NumPy, PyArrow |
| String Matching | RapidFuzz |
| Statistical Analysis | SciPy |

## Data

✅ **All required data is included in this repository** - no additional downloads needed!

- `dataset/processed/` - Pre-computed NASRI/ASRS scores and features
- `data_cache/india_districts.geojson` - District boundaries (594 districts)
- Maps are auto-generated on first run

See [`dataset/README.md`](dataset/README.md) for data format specifications.

## Screenshots

*Dashboard with NASRI choropleth map showing district-level readiness scores*

## Documentation

- [SETUP.md](SETUP.md) - Detailed setup instructions for all platforms
- [docs/MAP_OPTIMIZATION.md](docs/MAP_OPTIMIZATION.md) - Map performance optimizations
- [docs/LOCATION_CLEANER_GUIDE.md](docs/LOCATION_CLEANER_GUIDE.md) - Data cleaning documentation

## Project Team

**UIDAI Hackathon 2026**

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*Built for UIDAI Hackathon 2026* 🇮🇳
