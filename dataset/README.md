# Dataset Directory

This directory contains processed data files required by the ASEWIS application.

✅ **All processed data is included** - the dashboard works out-of-the-box!

## Structure

```
dataset/
├── processed/           # ✅ INCLUDED - Pre-processed parquet files
│   ├── aggregated_metrics.parquet
│   ├── anomalies.parquet
│   ├── features.parquet
│   ├── scores.parquet
│   └── unique_districts.txt
└── external/            # External reference data
```

## Regenerating Raw Data (Optional)

The raw data files (`dataset/raw/` and `dataset/api_data_*/`) are not included. To regenerate from fresh data:

### 1. Download Raw Aadhar Data

Raw data can be obtained from the UIDAI Open Data Portal:
- **Biometric Data**: https://uidai.gov.in/en/ecosystem/authentication-ecosystem/aadhaar-statistics.html
- **Demographic Data**: UIDAI monthly authentication reports
- **Enrolment Data**: State-wise enrolment statistics

### 2. Place Files in Correct Directories

```
dataset/
├── raw/                          # Raw CSV/Excel files
├── api_data_aadhar_biometric/    # Biometric authentication data
├── api_data_aadhar_demographic/  # Demographic update data
└── api_data_aadhar_enrolment/    # Enrolment statistics
```

### 3. Run the Data Pipeline

After placing raw data, regenerate processed files:

```bash
# Activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the data processing pipeline
python src/run_pipeline.py
```

This will:
1. Clean and validate raw data
2. Compute district-level metrics
3. Generate aggregated parquet files in `dataset/processed/`
4. Build map cache files in `data_cache/maps/`

## Processed Files Description

| File | Description |
|------|-------------|
| `aggregated_metrics.parquet` | Monthly aggregated metrics per district |
| `anomalies.parquet` | Detected anomalies and outliers |
| `features.parquet` | Engineered features for scoring |
| `scores.parquet` | NASRI and ASRS scores per district |
| `unique_districts.txt` | List of all recognized district names |

## Notes

- Processed files are sufficient to run the Streamlit dashboard
- Raw data is only needed if you want to update the processed files with new data
- The `data_cache/` directory contains GeoJSON and pre-built map visualizations
