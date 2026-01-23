# Ment_helt_AI 🧠

AI-powered Mental Health Support Chatbot with Retrieval-Augmented Generation (RAG)

## Overview

This project provides a conversational AI agent designed to offer supportive interactions for mental health topics. It uses:

- **GPT-2 Medium** for natural language generation
- **Local FAISS Vector Database** for retrieving relevant mental health information
- **Sentiment Analysis** for assessing user emotional state
- **Safety Features** including crisis detection and hotline recommendations
- **Streamlit** for an intuitive web interface

⚠️ **Important**: This is a supportive tool, NOT a replacement for professional mental health care.

## Quick Start

### For PyCharm Users 🚀

If you're using PyCharm IDE, we've set up everything you need! See the [**PyCharm Development Guide**](PYCHARM_GUIDE.md) for detailed instructions.

**TL;DR:**
1. Open the project in PyCharm
2. Run: `python setup.py`
3. Configure the Python interpreter (Settings > Project > Python Interpreter)
4. Click the green play button with "Streamlit App (module)" selected

### Standard Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sparsxlogan96/Ment_helt_AI.git
   cd Ment_helt_AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to http://localhost:8501

## Features

✅ **Conversational AI** - Natural dialogue using GPT-2  
✅ **RAG System** - Retrieves relevant mental health information from local knowledge base  
✅ **Sentiment Analysis** - Detects emotional state and adjusts responses  
✅ **Crisis Detection** - Identifies high-risk situations and provides crisis resources  
✅ **Privacy-First** - All data processing happens locally, no conversation data stored  
✅ **Customizable** - Fine-tune the model on your own mental health datasets  

## Project Structure

```
Ment_helt_AI/
├── .idea/                    # PyCharm configuration (auto-setup)
├── mental_health_data/       # Knowledge base for RAG
│   ├── anxiety.txt
│   ├── cbt_basics.txt
│   └── depression.txt
├── app.py                    # Main Streamlit application
├── fine_tune_gpt2.py        # Model fine-tuning script
├── setup.py                  # Development environment setup
├── requirements.txt          # Python dependencies
├── PYCHARM_GUIDE.md         # Detailed PyCharm setup guide
└── README.md                 # This file
```

## Customizing the Model

### Fine-Tuning GPT-2

Want to create your own custom GPT model trained on mental health data?

```bash
python fine_tune_gpt2.py \
  --data-dir mental_health_data \
  --output-dir ./fine_tuned_model \
  --epochs 5
```

Then update `app.py` to use your fine-tuned model:

```python
@st.cache_resource
def load_models():
    text_generator = pipeline("text-generation", model="./fine_tuned_model")
    sentiment_analyzer = pipeline("sentiment-analysis")
    return text_generator, sentiment_analyzer
```

See [PYCHARM_GUIDE.md](PYCHARM_GUIDE.md) for detailed instructions.

## Technologies Used

- **Streamlit** - Web interface
- **Transformers (Hugging Face)** - GPT-2 and sentiment analysis models
- **FAISS** - Vector similarity search for RAG
- **Sentence Transformers** - Text embeddings
- **PyTorch** - Deep learning framework

## Development

### Running Tests

```bash
# Coming soon - test infrastructure
```

### Code Quality

The project includes PyCharm inspection profiles for:
- PEP 8 compliance
- Type hints
- Code quality checks

## Ethical Considerations

### Data Privacy and Security
- Local FAISS index stores all data on your machine
- No user conversations are stored beyond the session
- All processing happens locally - no data sent to external servers
- In deployment, server security is critical for protecting data

### Model Bias
- AI models may reflect biases in training data
- Mental health data should be carefully curated
- Responses should be reviewed for potential bias

### Limitations
- This is NOT a substitute for professional mental health care
- AI cannot provide diagnosis or treatment
- Crisis situations require human intervention

### Safety Features
- Keyword-based crisis detection
- Automatic crisis hotline recommendations
- Clear disclaimers about AI limitations

For full ethical considerations, see the comments in `app.py`.

## Crisis Resources

If you or someone you know is in crisis:

- **National Suicide Prevention Lifeline**: Call or text **988**
- **Crisis Text Line**: Text **HOME** to **741741**
- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is provided for educational and research purposes.

## Acknowledgments

- Mental health data sourced from public health resources
- Built with Hugging Face Transformers and Streamlit
- Community feedback and contributions

---

**Remember**: If you're experiencing a mental health crisis, please contact a professional or crisis hotline immediately. This AI is a supportive tool, not a replacement for professional care.
