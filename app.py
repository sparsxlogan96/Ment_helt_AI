import streamlit as st
from transformers import pipeline
import logging
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os # Import os for file paths

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO)

# --- Model and Pipeline Initialization ---
# Use Streamlit's caching to load models only once
@st.cache_resource
def load_models():
    """Loads the text generation and sentiment analysis models."""
    try:
        text_generator = pipeline("text-generation", model="microsoft/DialoGPT-small")
        sentiment_analyzer = pipeline("sentiment-analysis")
        logging.info("Models loaded successfully.")
        return text_generator, sentiment_analyzer
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        st.error("Could not load AI models. Please try again later.")
        return None, None

# Load models
text_generator, sentiment_analyzer = load_models()

# Check if models loaded successfully
if text_generator is None or sentiment_analyzer is None:
    st.stop() # Stop the application if models failed to load


# --- Load and preprocess mental health and CBT data ---
# This should ideally be done once and the index saved/loaded.
# For this consolidated script, we include the loading and preprocessing here.
@st.cache_resource
def load_and_preprocess_data(directory="mental_health_data"):
    """
    Reads text files from a directory and preprocesses their content.
    Cached to load data only once.
    """
    documents = []
    # Ensure the directory exists
    if not os.path.exists(directory):
        logging.warning(f"Data directory '{directory}' not found. No local data will be loaded for RAG.")
        return []

    filepaths = glob.glob(os.path.join(directory, "*.txt"))

    if not filepaths:
        logging.warning(f"No .txt files found in '{directory}'. No local data will be loaded for RAG.")
        return []

    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            content = re.sub(r'\s+', ' ', content).strip()
            content = content.lower()
            documents.append(content)
            logging.info(f"Loaded and preprocessed: {filepath}")
        except Exception as e:
            logging.error(f"Error reading or processing file {filepath}: {e}")

    return documents

# Load the documents
mental_health_documents = load_and_preprocess_data()


# --- Choose and implement an embedding model ---
@st.cache_resource
def load_embedding_model():
    """Loads the sentence transformer embedding model."""
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("Embedding model loaded successfully.")
        return embedding_model
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        return None

embedding_model = load_embedding_model()

if embedding_model is None:
    st.error("Could not load the embedding model. Please check your setup.")
    st.stop()


# --- Create and populate a local vector store (FAISS) ---
@st.cache_resource
def setup_faiss_index(_documents, _embedding_model):
    """Creates and populates a FAISS index from documents and embeddings."""
    if not _documents or _embedding_model is None:
        logging.warning("No documents or embedding model available to create FAISS index.")
        return None # Return None for index

    try:
        logging.info(f"Generating embeddings for {len(_documents)} documents for FAISS index...")
        document_embeddings = _embedding_model.encode(_documents).astype('float32')
        logging.info(f"Document embeddings shape: {document_embeddings.shape}")

        embedding_dimension = document_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dimension)
        logging.info(f"FAISS index created with dimension {embedding_dimension}.")

        index.add(document_embeddings)
        logging.info(f"Document embeddings added to the FAISS index. Total vectors in index: {index.ntotal}")

        return index

    except Exception as e:
        logging.error(f"Error setting up FAISS index: {e}")
        return None # Return None if setup fails


# Setup FAISS index (requires documents and embedding model)
# Adding a spinner while setting up FAISS
with st.spinner("Setting up local FAISS vector database..."):
    faiss_index = setup_faiss_index(mental_health_documents, embedding_model)


if faiss_index is None:
    st.warning("FAISS index could not be set up. RAG functionality will be disabled.")
    # Do not stop, allow the app to run without RAG if index setup fails


# --- Implement a local retrieval mechanism (FAISS) ---
def retrieve_info_local(query, index, embedding_model, documents, top_k=3):
    """
    Retrieves relevant information from the local FAISS index based on a user query.
    """
    if index is None or embedding_model is None or documents is None:
        logging.warning("FAISS index, embedding model, or documents not available for retrieval.")
        return []

    try:
        query_embedding = embedding_model.encode(query).astype('float32')
        query_embedding = np.array([query_embedding]) # FAISS expects 2D array

        distances, indices = index.search(query_embedding, top_k)

        retrieved_documents = []
        for doc_idx in indices[0]:
            if 0 <= doc_idx < len(documents):
                retrieved_documents.append(documents[doc_idx])
            else:
                logging.warning(f"Retrieved invalid document index {doc_idx}.")

        return retrieved_documents

    except Exception as e:
        logging.error(f"Error during local information retrieval: {e}")
        return []


# --- AI Agent Functions (Modified for RAG and Streamlit) ---

def generate_response(user_input, history=None, retrieved_info=None):
    """
    Generates a response using the loaded text generation model, incorporating retrieved information.

    Args:
        user_input (str): The user's current input.
        history (str, optional): The conversation history. Defaults to "".
        retrieved_info (list, optional): A list of strings containing retrieved relevant documents. Defaults to None.

    Returns:
        tuple: A tuple containing the generated response text (str) and the updated conversation history (str).
    """
    if text_generator is None:
         logging.error("Text generator model not loaded.")
         return "I'm sorry, the conversation model is not available.", history

    if history is None:
        history = ""

    # Construct the input prompt for the LLM, incorporating retrieved information
    prompt_parts = []

    # Add retrieved information to the prompt if available
    if retrieved_info:
        prompt_parts.append("Context from mental health resources:")
        for i, doc in enumerate(retrieved_info):
            prompt_parts.append(f"Document {i+1}: {doc}")
        prompt_parts.append("\nBased on the above information and our conversation:")

    # Add conversation history and current user input
    # Using the "User: ... Bot: ..." format that DialoGPT understands
    formatted_history = history.strip() + text_generator.tokenizer.eos_token if history else ""

    prompt_parts.append(formatted_history)
    prompt_parts.append("User: " + user_input + text_generator.tokenizer.eos_token)
    prompt_parts.append("Bot: ") # Marker for bot's turn


    full_input = "".join(prompt_parts)

    try:
        generated_sequence = text_generator(
            full_input,
            max_length=len(full_input) + 150,  # Generate up to 150 new tokens
            pad_token_id=text_generator.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            num_return_sequences=1,
            truncation=True
        )

        generated_text = generated_sequence[0]['generated_text']

        # Extract the generated text that comes AFTER the last "Bot: " marker
        bot_marker = "Bot: "
        response_start_index = generated_text.rfind(bot_marker, 0, len(full_input)) + len(bot_marker)
        response = generated_text[response_start_index:].strip()

        # Clean up the extracted response
        response = response.replace(text_generator.tokenizer.eos_token, "").strip()
        turn_end_markers = ["User:", "Bot:"]
        shortest_end_index = len(response)
        for marker in turn_end_markers:
             marker_index = response.find(marker)
             if marker_index != -1 and (marker_index == 0 or response[marker_index-1].isspace()):
                 if marker_index < shortest_end_index:
                     shortest_end_index = marker_index
        response = response[:shortest_end_index].strip()

        # Basic safety check
        unsafe_keywords = ["kill", "suicide", "harm yourself", "end my life", "unalive myself"]
        if any(keyword in response.lower() for keyword in unsafe_keywords):
            logging.warning("Potential unsafe response detected.")
            response = "I'm here to support you. Remember, if you are having thoughts of harming yourself, please reach out to a crisis hotline or mental health professional immediately."

        # Update history
        updated_history = history + "User: " + user_input + text_generator.tokenizer.eos_token + "Bot: " + response + text_generator.tokenizer.eos_token

        return response, updated_history

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, I encountered an error while generating a response. Could you please rephrase that?", history


def assess_severity(user_input, _sentiment_analyzer):
    """
    Assesses the severity of the user's mental health state based on input text.
    """
    if _sentiment_analyzer is None:
        logging.warning("Sentiment analyzer not loaded, cannot assess severity.")
        return 'low'

    high_severity_keywords = ["suicide", "kill myself", "ending it all", "hopeless", "no point",
                              "end my life", "harm myself", "want to die", "can't go on", "unalive myself"]
    medium_severity_keywords = ["sad", "depressed", "anxious", "stressed", "struggling", "down",
                                "overwhelmed", "lonely", "empty", "worried", "difficult"]

    user_input_lower = user_input.lower()
    for keyword in high_severity_keywords:
        if keyword in user_input_lower:
            return 'high'

    try:
        sentiment_result = _sentiment_analyzer(user_input)[0]
        sentiment_score = sentiment_result['score']
        sentiment_label = sentiment_result['label']

        if sentiment_label == 'NEGATIVE' and sentiment_score > 0.7:
            for keyword in medium_severity_keywords:
                if keyword in user_input_lower:
                    return 'medium'
            return 'medium'
        elif sentiment_label == 'NEGATIVE' and sentiment_score > 0.4:
             return 'medium'
        else:
            return 'low'
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return 'low'


def provide_recommendations(severity_level):
    """
    Provides recommendations based on the assessed severity level.
    """
    if severity_level == 'high':
        return "Your situation sounds serious. Please seek immediate professional help or contact a crisis hotline. **You can call the National Suicide Prevention Lifeline at 988 or text HOME to 741741 to reach the Crisis Text Line.** Your safety is the top priority."
    elif severity_level == 'medium':
        return "It sounds like you are going through a challenging time. Talking to someone or exploring resources might be helpful."
    else: # severity_level == 'low'
        return "It's good that you are reaching out. Focusing on self-care can be beneficial. Try practicing mindfulness, getting enough sleep, exercising, and connecting with friends and family."


# --- Streamlit UI ---

st.title("Mental Health Support AI Agent (Local RAG Enabled)") # Updated title for local RAG

# Add a prominent disclaimer
st.warning("""
**Disclaimer:** This AI agent is for informational and conversational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing a mental health crisis or have serious concerns about your well-being, please consult a qualified healthcare provider or contact a crisis hotline immediately.
""")


# Initialize conversation history and messages list in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = ""

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is on your mind today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assess the severity of the user's input immediately
    severity = assess_severity(prompt, sentiment_analyzer) # Pass sentiment_analyzer

    # Handle high severity input specifically
    if severity == 'high':
        recommendations = provide_recommendations(severity)
        full_response = "Your input indicates a high level of distress. " + recommendations
        with st.chat_message("assistant"):
            st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # Process normal input
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # --- Local RAG Logic (using FAISS) ---
                retrieved_docs = []
                # Check if FAISS index, embedding model, and documents are available
                if faiss_index is not None and embedding_model is not None and mental_health_documents:
                     # Retrieve relevant information from the FAISS index
                     retrieved_docs = retrieve_info_local(prompt, faiss_index, embedding_model, mental_health_documents)
                     if retrieved_docs:
                         logging.info(f"Retrieved {len(retrieved_docs)} documents from FAISS.")
                         # Optional: Display retrieved docs for debugging
                         # st.write("Retrieved Info:")
                         # for doc in retrieved_docs:
                         #      st.write(f"- {doc[:100]}...") # Display first 100 chars
                     else:
                         logging.info("No relevant documents retrieved from FAISS.")
                else:
                    logging.warning("FAISS index, embedding model, or documents not available. Skipping RAG retrieval.")


                # Generate response using the LLM, incorporating retrieved information
                response_text, st.session_state.conversation_history = generate_response(
                    prompt,
                    st.session_state.conversation_history,
                    retrieved_info=retrieved_docs # Pass retrieved documents to generate_response
                )

                # Provide recommendations based on severity (already assessed)
                recommendations = provide_recommendations(severity)

                # Combine response and recommendations ONLY if severity is not low
                if severity != 'low':
                     full_response = response_text + "\n\n" + recommendations
                else:
                     full_response = response_text # For low severity, just the generated response

                st.markdown(full_response)

        # Add assistant's full response (including recommendations) to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Note: To run this Streamlit application locally:
# 1. Save this code as a .py file (e.g., app.py).
# 2. Ensure you have a 'mental_health_data' directory with .txt files containing your data
#    in the same directory as the app.py file.
# 3. Open your terminal in that directory.
# 4. Run the command: `streamlit run app.py`
