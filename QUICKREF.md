# Quick Reference Card 📝

## Running the App

### PyCharm
1. Select "Streamlit App (module)" from run configs dropdown
2. Click green ▶️ button
3. Open http://localhost:8501

### Command Line
```bash
streamlit run app.py
```

## Fine-Tuning the Model

```bash
# Basic fine-tuning
python fine_tune_gpt2.py

# Custom parameters
python fine_tune_gpt2.py \
  --model-name gpt2-medium \
  --data-dir mental_health_data \
  --output-dir ./fine_tuned_model \
  --epochs 5 \
  --batch-size 2
```

## Project Setup

### First Time Setup
```bash
python setup.py
```

### Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `fine_tune_gpt2.py` | Model customization script |
| `setup.py` | Environment setup automation |
| `PYCHARM_GUIDE.md` | Detailed PyCharm instructions |
| `mental_health_data/` | RAG knowledge base |
| `.idea/` | PyCharm configuration |

## Dependencies

- streamlit - Web UI framework
- transformers - Hugging Face models (GPT-2)
- sentence-transformers - Text embeddings
- faiss-cpu - Vector similarity search
- torch - Deep learning backend
- numpy - Numerical operations

## Useful Commands

```bash
# Check Python version
python --version

# List installed packages
pip list

# Update a package
pip install --upgrade <package-name>

# Freeze dependencies
pip freeze > requirements.txt

# Run on different port
streamlit run app.py --server.port 8502
```

## PyCharm Shortcuts

| Action | Windows/Linux | Mac |
|--------|---------------|-----|
| Run | Shift+F10 | Ctrl+R |
| Debug | Shift+F9 | Ctrl+D |
| Search everywhere | 2x Shift | 2x Shift |
| Go to definition | Ctrl+B | Cmd+B |
| Find usages | Alt+F7 | Opt+F7 |
| Recent files | Ctrl+E | Cmd+E |
| Terminal | Alt+F12 | Opt+F12 |

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### Port in Use
```bash
streamlit run app.py --server.port 8502
```

### Models Not Loading
- First run downloads models (~1.5GB for gpt2-medium)
- Subsequent runs use cached models
- Check internet connection for first run

### FAISS Issues on Windows
```bash
conda install faiss-cpu -c pytorch
```

## Resources

- 📚 [PYCHARM_GUIDE.md](PYCHARM_GUIDE.md) - Full PyCharm setup
- 📖 [README.md](README.md) - Project overview
- 🌐 [Streamlit Docs](https://docs.streamlit.io/)
- 🤗 [Transformers Docs](https://huggingface.co/docs/transformers/)

## Crisis Resources

**If you need immediate help:**
- Call or text **988** (US Suicide Prevention Lifeline)
- Text **HOME** to **741741** (Crisis Text Line)

---

💡 **Tip**: Keep this file open in PyCharm for quick reference!
