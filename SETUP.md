# ASEWIS Setup Guide

Complete setup instructions for Windows, macOS, and Linux.

---

## Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- **pip** (usually comes with Python)
- **Git** (optional, for version control)

Check your Python version:
```bash
python --version
```

---

## Setup Instructions

### 1. Create Virtual Environment

**Windows (PowerShell):**
```powershell
# Navigate to project directory
cd "d:\HACKATHONS\UIDAI Hackathon\asewis"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
cd "d:\HACKATHONS\UIDAI Hackathon\asewis"
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux (bash/zsh):**
```bash
# Navigate to project directory
cd ~/HACKATHONS/UIDAI\ Hackathon/asewis

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**Verify activation:**
After activation, your prompt should show `(venv)` prefix:
```
(venv) PS D:\HACKATHONS\UIDAI Hackathon\asewis>
```

---

### 2. Upgrade pip (Recommended)

**All platforms:**
```bash
python -m pip install --upgrade pip
```

---

### 3. Install Dependencies

**Standard installation (all platforms):**
```bash
pip install -r requirements.txt
```

**Alternative: Install with verbose output (for debugging):**
```bash
pip install -r requirements.txt -v
```

**Alternative: Install from cache (faster for reinstalls):**
```bash
pip install --no-index --find-links=./wheels -r requirements.txt
```

**Verify installation:**
```bash
pip list
```

Expected output should include:
- pandas (2.1.4)
- numpy (1.26.3)
- streamlit (1.29.0)
- pyarrow (14.0.2)
- matplotlib (3.8.2)
- plotly (5.18.0)
- scipy (1.11.4)

---

### 4. Verify Setup

**Test imports:**
```bash
python -c "import pandas, numpy, streamlit, pyarrow, matplotlib, plotly, scipy; print('✅ All dependencies installed successfully')"
```

**Check Streamlit:**
```bash
streamlit --version
```

Expected output:
```
Streamlit, version 1.29.0
```

---

## Running the Application

### Option 1: Run Streamlit Dashboard

```bash
# From project root
streamlit run app/streamlit_app.py
```

The dashboard will open in your browser at: `http://localhost:8501`

### Option 2: Run Data Pipeline

```bash
# Navigate to src directory
cd src

# Step 1: Load and aggregate data
python aggregation.py

# Step 2: Compute features
python feature_engineering.py

# Step 3: Compute scores
python scoring.py

# Step 4: Generate recommendations
python rules_engine.py

# Step 5: Run simulations
python simulation.py
```

---

## Troubleshooting

### Issue: PowerShell Execution Policy Error

**Error message:**
```
.\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.
```

**Solution:**
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again:
```powershell
.\venv\Scripts\Activate.ps1
```

---

### Issue: Python not found

**Windows:**
- Ensure Python is in PATH
- Try `py` instead of `python`
- Or use full path: `C:\Python311\python.exe`

**macOS/Linux:**
- Try `python3` instead of `python`
- Install via: `brew install python` (macOS) or `apt-get install python3` (Linux)

---

### Issue: pip install fails with SSL errors

**Solution:**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

---

### Issue: Conflicting dependencies

**Solution: Clean reinstall**
```bash
# Deactivate and delete venv
deactivate
rm -rf venv  # macOS/Linux
# or
Remove-Item -Recurse -Force venv  # Windows PowerShell

# Create fresh environment
python -m venv venv
# Activate (see activation commands above)
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Issue: Out of memory during data processing

**Solution: Increase chunk size in data_loading.py**
```python
# Edit src/data_loading.py
CHUNK_SIZE = 50000  # Reduce from 100000
```

---

## Deactivating Virtual Environment

**All platforms:**
```bash
deactivate
```

---

## Development Workflow

### Quick Start (after initial setup):

**Windows:**
```powershell
cd "d:\HACKATHONS\UIDAI Hackathon\asewis"
.\venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

**macOS/Linux:**
```bash
cd ~/HACKATHONS/UIDAI\ Hackathon/asewis
source venv/bin/activate
streamlit run app/streamlit_app.py
```

---

## Production Deployment Notes

### For Streamlit Cloud:

1. Ensure `requirements.txt` is in project root
2. Create `.streamlit/config.toml`:
   ```toml
   [server]
   maxUploadSize = 500
   enableCORS = false
   
   [browser]
   gatherUsageStats = false
   ```

3. Deploy via: https://streamlit.io/cloud

### For Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Performance Tips

1. **Use Parquet for large datasets** (already implemented)
2. **Enable Streamlit caching** (already implemented with `@st.cache_data`)
3. **Process data in chunks** (already implemented in data_loading.py)
4. **Monitor memory usage:**
   ```bash
   python -m memory_profiler src/aggregation.py
   ```

---

## License

This project is for UIDAI Hackathon 2026.

---

## Support

For issues during hackathon:
1. Check this troubleshooting guide
2. Verify Python version: `python --version`
3. Verify all dependencies: `pip list`
4. Check logs in `logs/` directory

---

**Last Updated:** January 8, 2026
