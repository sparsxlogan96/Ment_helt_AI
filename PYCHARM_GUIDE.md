# PyCharm Development Guide

This guide helps you set up and work with the Mental Health AI project in PyCharm IDE.

## Quick Start

### 1. Open Project in PyCharm

1. Launch PyCharm
2. Click `File` > `Open`
3. Navigate to and select the `Ment_helt_AI` directory
4. Click `OK`

PyCharm will automatically detect the project configuration files and set up the project structure.

### 2. Set Up Python Interpreter

#### Option A: Automatic Setup (Recommended)

Run the setup script to automatically create a virtual environment and install dependencies:

```bash
python setup.py
```

Then configure PyCharm to use the created virtual environment:
1. Go to `Settings/Preferences` > `Project: Ment_helt_AI` > `Python Interpreter`
2. Click the gear icon ⚙️ > `Add`
3. Select `Existing environment`
4. Navigate to: `<project-directory>/venv/bin/python` (or `venv\Scripts\python.exe` on Windows)
5. Click `OK`

#### Option B: Manual Setup in PyCharm

1. Go to `Settings/Preferences` > `Project: Ment_helt_AI` > `Python Interpreter`
2. Click the gear icon ⚙️ > `Add`
3. Select `Virtualenv Environment` > `New environment`
4. Choose a location for the virtual environment (default is fine)
5. Click `OK`
6. Open the terminal in PyCharm and run:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Run the Application

#### Using Run Configuration (Easiest)

The project comes with pre-configured run settings:

1. Look for the run configuration dropdown in the top toolbar
2. Select `Streamlit App (module)`
3. Click the green play button ▶️
4. The app will start and PyCharm will show the URL in the console

#### Using Terminal

1. Open the terminal in PyCharm (`View` > `Tool Windows` > `Terminal`)
2. Make sure your virtual environment is activated (should be automatic in PyCharm)
3. Run:
   ```bash
   streamlit run app.py
   ```

#### Manual Python Console

You can also run it from Python console:
```python
import subprocess
subprocess.run(["streamlit", "run", "app.py"])
```

### 4. Access the Application

Once running, open your web browser to:
- **Local URL**: http://localhost:8501
- **Network URL**: http://[your-ip]:8501

## Project Structure

```
Ment_helt_AI/
├── .idea/                      # PyCharm project configuration
│   ├── runConfigurations/      # Pre-configured run settings
│   ├── inspectionProfiles/     # Code inspection settings
│   ├── codeStyles/             # Code style settings
│   └── ...
├── mental_health_data/         # Training data for RAG
│   ├── anxiety.txt
│   ├── cbt_basics.txt
│   └── depression.txt
├── app.py                      # Main Streamlit application
├── fine_tune_gpt2.py          # Script to fine-tune GPT-2 model
├── setup.py                    # Development setup script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Working with the Code

### Code Navigation Tips

- **Go to Definition**: `Ctrl+B` (Windows/Linux) or `Cmd+B` (Mac)
- **Find Usages**: `Alt+F7` (Windows/Linux) or `Opt+F7` (Mac)
- **Search Everywhere**: Double press `Shift`
- **Recent Files**: `Ctrl+E` (Windows/Linux) or `Cmd+E` (Mac)

### Debugging the Application

1. Set breakpoints by clicking in the gutter (left of line numbers)
2. Instead of using the run configuration, use the debug button (🐞)
3. PyCharm will pause at breakpoints and show variable values

### Code Quality

The project includes code inspection profiles that check for:
- PEP 8 compliance (with sensible exceptions)
- Type hints
- Unused imports
- Naming conventions

To run inspections:
- `Code` > `Inspect Code`
- Or use `Ctrl+Alt+Shift+I` (Windows/Linux) or `Cmd+Opt+Shift+I` (Mac)

## Customizing the GPT Model

### Fine-Tuning GPT-2

To create your own custom GPT-2 model trained on your mental health data:

1. **Prepare your data**:
   - Add more `.txt` files to the `mental_health_data/` directory
   - Each file should contain relevant mental health information

2. **Run the fine-tuning script**:
   ```bash
   python fine_tune_gpt2.py --data-dir mental_health_data --output-dir ./fine_tuned_model --epochs 5
   ```

3. **Parameters you can adjust**:
   - `--model-name`: Base model (gpt2, gpt2-medium, gpt2-large)
   - `--data-dir`: Directory with training data
   - `--output-dir`: Where to save the fine-tuned model
   - `--epochs`: Number of training epochs (more = better fit, but risk overfitting)
   - `--batch-size`: Training batch size (adjust based on your GPU memory)
   - `--block-size`: Maximum sequence length

4. **Use the fine-tuned model** in your app:
   
   Edit `app.py` and modify the model loading section:
   
   ```python
   @st.cache_resource
   def load_models():
       # Use your fine-tuned model instead of the default
       text_generator = pipeline("text-generation", model="./fine_tuned_model")
       sentiment_analyzer = pipeline("sentiment-analysis")
       return text_generator, sentiment_analyzer
   ```

### Training Considerations

- **Data Quality**: More high-quality mental health data = better responses
- **Computational Resources**: Fine-tuning requires significant GPU/CPU time
- **Epochs**: Start with 3-5 epochs, monitor loss to avoid overfitting
- **Evaluation**: Test the model thoroughly before using in production

## Troubleshooting

### Common Issues

#### 1. Module Not Found Errors

**Problem**: Import errors like `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```bash
pip install -r requirements.txt
```

Make sure PyCharm is using the correct interpreter with packages installed.

#### 2. FAISS Installation Issues

**Problem**: Error installing `faiss-cpu`

**Solution**:
- On Windows: Use conda instead: `conda install faiss-cpu -c pytorch`
- Or try: `pip install faiss-cpu --no-cache-dir`

#### 3. Streamlit Not Found

**Problem**: `streamlit: command not found`

**Solution**:
```bash
pip install streamlit
# Or run using module syntax
python -m streamlit run app.py
```

#### 4. Port Already in Use

**Problem**: "Address already in use" error

**Solution**:
```bash
# Run on a different port
streamlit run app.py --server.port 8502
```

#### 5. Models Taking Too Long to Load

**Problem**: First run is very slow

**Solution**: This is normal! The models are being downloaded:
- `gpt2-medium`: ~1.5 GB
- `all-MiniLM-L6-v2`: ~90 MB
- Subsequent runs will be much faster (models are cached)

### Getting Help

If you encounter issues:

1. Check the PyCharm Event Log (bottom-right corner)
2. Review the terminal output for error messages
3. Verify all dependencies are installed: `pip list`
4. Try running the setup script again: `python setup.py`

## Advanced Features

### Using PyCharm's Built-in Tools

#### 1. Scientific Mode

For data analysis and model experimentation:
- View > Scientific Mode
- Run code cells interactively
- Visualize data frames

#### 2. Python Console

For quick testing:
- Tools > Python Console
- Test functions without running the whole app

#### 3. TODO Comments

Mark areas for improvement:
```python
# TODO: Add more sophisticated sentiment analysis
# FIXME: Handle edge case when no documents are retrieved
```

View all TODOs: View > Tool Windows > TODO

#### 4. Version Control Integration

PyCharm has built-in Git support:
- VCS menu for git operations
- Commit tool window (Ctrl+K / Cmd+K)
- View changes and diff files easily

### Performance Tips

1. **Enable power save mode** when not actively developing to reduce CPU usage
2. **Exclude directories** from indexing:
   - Right-click on `venv/`, `.idea/` folders
   - Mark Directory as > Excluded
3. **Increase memory** for PyCharm (Help > Change Memory Settings)

## Best Practices

1. **Use Virtual Environments**: Always work in a virtual environment to isolate dependencies
2. **Type Hints**: Add type hints to functions for better code completion
3. **Docstrings**: Document your functions with docstrings
4. **Git Commits**: Make small, frequent commits with clear messages
5. **Code Reviews**: Use PyCharm's compare feature before committing

## Next Steps

- Explore the `app.py` file to understand the application flow
- Review the mental health data files in `mental_health_data/`
- Try fine-tuning the model with your own data
- Experiment with different GPT-2 model sizes
- Add more features to the Streamlit interface

## Resources

- [PyCharm Documentation](https://www.jetbrains.com/pycharm/learn/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [GPT-2 Model Card](https://huggingface.co/gpt2-medium)

---

Happy coding! 🚀
