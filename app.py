# main_app.py (Consolidated Code)

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM # Import necessary classes
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import glob
import re
import logging
from google.colab import userdata # Used only in Colab for API key access

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO)

# --- Load and preprocess mental health and CBT data ---
def load_and_preprocess_data(directory):
    """
    Reads text files from a directory and preprocesses their content.

    Args:
        directory (str): The path to the directory containing the text files.

    Returns:
        list: A list of strings, where each string is the preprocessed content of a file.
    """
    documents = []
    filepaths = glob.glob(os.path.join(directory, "*.txt")) # Read all .txt files

    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Preprocessing steps:
            # Remove extra whitespace (including newlines and tabs)
            content = re.sub(r'\s+', ' ', content).strip()
            # Convert to lowercase
            content = content.lower()
            # Optional: Remove punctuation (depending on embedding model)
            # content = re.sub(r'[^\w\s]', '', content)

            documents.append(content)
            logging.info(f"Loaded and preprocessed: {filepath}")
        except Exception as e:
            logging.error(f"Error reading or processing file {filepath}: {e}")

    return documents

# --- Choose and implement an embedding model ---
@st.cache_resource # Cache the embedding model
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


# --- Set up a vector database (Pinecone) ---
@st.cache_resource # Cache the Pinecone connection and index
def setup_pinecone():
    """Initializes Pinecone and connects to the index."""
    try:
        # In a deployed environment, get API key from environment variables or secrets
        # In Colab, we use userdata.get('PINECONE_API_KEY')
        # Replace with os.environ.get('PINECONE_API_KEY') in deployment
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') or userdata.get('PINECONE_API_KEY')


        if not PINECONE_API_KEY:
             raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable or Colab secret.")

        pc = Pinecone(api_key=PINECONE_API_KEY)
        logging.info("Pinecone initialized successfully!")

        index_name = "mental-health-rag"
        embedding_dimension = 384 # Dimension of 'all-MiniLM-L6-v2'

        if index_name not in pc.list_indexes().names:
             logging.info(f"Creating index '{index_name}'...")
             pc.create_index(
                name=index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1") # Adjust as needed
            )
             logging.info(f"Index '{index_name}' created.")
        else:
             logging.info(f"Index '{index_name}' already exists.")

        index = pc.Index(index_name)
        logging.info(f"Connected to index '{index_name}'.")
        return index

    except Exception as e:
        logging.error(f"Error setting up Pinecone: {e}")
        return None

index = setup_pinecone()

if index is None:
    st.error("Could not connect to Pinecone index. Please check your API key and index details.")
    st.stop()

# --- Implement the retrieval mechanism ---
def retrieve_info(query, index, embedding_model, top_k=3):
    """
    Retrieves relevant information from the Pinecone index based on a user query.
    """
    if index is None or embedding_model is None:
        logging.warning("Pinecone index or embedding model not available for retrieval.")
        return []

    try:
        query_embedding = embedding_model.encode(query).tolist()

        search_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        retrieved_documents = []
        if search_results and search_results.matches:
            for match in search_results.matches:
                if 'metadata' in match and 'text' in match['metadata']:
                    retrieved_documents.append(match['metadata']['text'])

        return retrieved_documents

    except Exception as e:
        logging.error(f"Error during information retrieval: {e}")
        return []


# --- AI Agent Functions (Modified for RAG and Streamlit) ---

# Cache the sentiment analyzer
@st.cache_resource
def load_sentiment_analyzer():
    """Loads the sentiment analysis model."""
    try:
        sentiment_analyzer = pipeline("sentiment-analysis")
        logging.info("Sentiment analyzer loaded successfully.")
        return sentiment_analyzer
    except Exception as e:
        logging.error(f"Error loading sentiment analyzer: {e}")
        return None

sentiment_analyzer = load_sentiment_analyzer()

if sentiment_analyzer is None:
    st.error("Could not load the sentiment analysis model. Please check your setup.")
    st.stop()


# Cache the text generation model
@st.cache_resource
def load_text_generator():
    """Loads the text generation model."""
    try:
        # Using a smaller model for potentially faster responses in deployment
        text_generator = pipeline("text-generation", model="microsoft/DialoGPT-small")
        logging.info("Text generation model loaded successfully.")
        return text_generator
    except Exception as e:
        logging.error(f"Error loading text generation model: {e}")
        return None

text_generator = load_text_generator()

if text_generator is None:
    st.error("Could not load the text generation model. Please check your setup.")
    st.stop()


def generate_response(user_input, history=None, retrieved_info=None):
    """
    Generates a response using the loaded text generation model, incorporating retrieved information.
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
        for doc in retrieved_info:
            prompt_parts.append(doc)
        prompt_parts.append("\nBased on the above information and our conversation:")

    # Add conversation history and current user input
    # We need to format the history appropriately for the LLM (e.g., turn-by-turn)
    # For DialoGPT, appending with EOS tokens is suitable.
    prompt_parts.append(history)
    prompt_parts.append("User: " + user_input + text_generator.tokenizer.eos_token) # Add current user input

    full_input = "".join(prompt_parts)

    try:
        generated_sequence = text_generator(
            full_input,
            max_length=len(full_input) + 100,  # Generate up to 100 new tokens (increased for potentially longer RAG responses)
            pad_token_id=text_generator.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=1,
            truncation=True # Truncate input if it's too long
        )

        generated_text = generated_sequence[0]['generated_text']

        # Extract the generated text. The output includes the input prompt, so we need to remove it.
        # We also need to remove the EOS token if it's at the end.
        # Find the index of the first occurrence of the EOS token after the input
        # Need to be careful here, the prompt includes history and potentially context
        # A simpler approach for DialoGPT is to find the response after the last EOS token from the input
        response_start_index = generated_text.rfind(text_generator.tokenizer.eos_token) + len(text_generator.tokenizer.eos_token)
        response = generated_text[response_start_index:].strip()

        # Basic safety check on the generated response
        unsafe_keywords = ["kill", "suicide", "harm yourself", "end my life", "unalive myself"] # Add more keywords as needed
        if any(keyword in response.lower() for keyword in unsafe_keywords):
            logging.warning("Potential unsafe response detected.")
            response = "I'm here to support you. Remember, if you are having thoughts of harming yourself, please reach out to a crisis hotline or mental health professional immediately."
            # In a real application, you might want more sophisticated filtering or moderation

        # Update history for the next turn
        # Append the user input and the generated response to the history
        updated_history = history + "User: " + user_input + text_generator.tokenizer.eos_token + "Bot: " + response + text_generator.tokenizer.eos_token

        return response, updated_history

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, I encountered an error while generating a response. Could you please rephrase that?", history # Return a graceful error message


def assess_severity(user_input):
    """
    Assesses the severity of the user's mental health state based on input text.
    Enhanced to be more sensitive to high severity keywords.
    """
    if sentiment_analyzer is None:
        logging.warning("Sentiment analyzer not loaded, cannot assess severity.")
        return 'low' # Cannot assess if analyzer not loaded

    # Define keywords indicating higher severity - include variations
    high_severity_keywords = ["suicide", "kill myself", "ending it all", "hopeless", "no point",
                              "end my life", "harm myself", "want to die", "can't go on", "unalive myself"] # Added "unalive myself"
    medium_severity_keywords = ["sad", "depressed", "anxious", "stressed", "struggling", "down",
                                "overwhelmed", "lonely", "empty", "worried", "difficult"] # Added "worried", "difficult"

    # Check for high severity keywords - immediate flag
    user_input_lower = user_input.lower()
    for keyword in high_severity_keywords:
        if keyword in user_input_lower:
            return 'high'

    try:
        # Perform sentiment analysis only if no high severity keywords are found
        sentiment_result = sentiment_analyzer(user_input)[0]
        sentiment_score = sentiment_result['score']
        sentiment_label = sentiment_result['label']

        # Check sentiment score and medium severity keywords
        if sentiment_label == 'NEGATIVE' and sentiment_score > 0.7:
            for keyword in medium_severity_keywords:
                if keyword in user_input_lower:
                    return 'medium'
            # If strong negative but no medium keywords, still consider it medium
            return 'medium'
        elif sentiment_label == 'NEGATIVE' and sentiment_score > 0.4:
             # Moderate negative sentiment
             return 'medium'
        else:
            # Neutral or positive sentiment, or weak negative without keywords
            return 'low'
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return 'low' # Default to low severity in case of error


def provide_recommendations(severity_level):
    """
    Provides recommendations based on the assessed severity level.
    Includes specific crisis information for high severity.
    """
    if severity_level == 'high':
        return "Your situation sounds serious. Please seek immediate professional help or contact a crisis hotline. **You can call the National Suicide Prevention Lifeline at 988 or text HOME to 741741 to reach the Crisis Text Line.** Your safety is the top priority."
    elif severity_level == 'medium':
        return "It sounds like you are going through a challenging time. Talking to a counselor or therapist could be helpful. You can look for mental health professionals in your area or explore online therapy options."
    else: # severity_level == 'low'
        return "It's good that you are reaching out. For low severity concerns, focusing on self-care can be beneficial. Try practicing mindfulness, getting enough sleep, exercising, and connecting with friends and family."


# --- Streamlit UI ---

st.title("Mental Health Support AI Agent")

# Add a prominent disclaimer at the beginning
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
    severity = assess_severity(prompt)

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
                # --- RAG Integration ---
                # Retrieve relevant information based on the user's prompt
                retrieved_docs = retrieve_info(prompt, index, embedding_model)
                if retrieved_docs:
                    logging.info(f"Retrieved {len(retrieved_docs)} documents.")
                    # print("Retrieved documents:", retrieved_docs) # For debugging
                else:
                    logging.info("No relevant documents retrieved.")

                # Generate response using the LLM, including retrieved information
                response_text, st.session_state.conversation_history = generate_response(
                    prompt,
                    st.session_state.conversation_history,
                    retrieved_info=retrieved_docs # Pass retrieved documents to generate_response
                )

                # Provide recommendations based on severity (already assessed)
                recommendations = provide_recommendations(severity)

                # Combine response and recommendations (can refine how this is presented)
                # Only add recommendations if not low severity, or always add based on design choice
                if severity != 'low':
                     full_response = response_text + "\n\n" + recommendations
                else:
                     full_response = response_text # Don't clutter low severity with recommendations every time

                st.markdown(full_response)

        # Add assistant's full response (including recommendations) to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# Note: To run this Streamlit application, save the code as a .py file (e.g., app.py)
# and run 'streamlit run app.py' in your terminal.
# Ensure you have your Pinecone API key set as an environment variable or in Colab secrets.
# You will also need to create the 'mental_health_data' directory and add your text files there,
# then run a separate script (or add code to this script, run once) to embed and upsert
# the documents to Pinecone before running the Streamlit app for the first time.
