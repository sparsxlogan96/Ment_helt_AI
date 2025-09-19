import streamlit as st
from transformers import pipeline
import logging

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO)

# --- Model and Pipeline Initialization ---
# Use Streamlit's caching to load models only once
@st.cache_resource
def load_models():
    """Loads the text generation and sentiment analysis models."""
    try:
        # Changed model to gpt2-medium as requested
        text_generator = pipeline("text-generation", model="gpt2-medium")
        sentiment_analyzer = pipeline("sentiment-analysis")
        logging.info("Models loaded successfully.")
        return text_generator, sentiment_analyzer
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        st.error("Could not load AI models. Please try again later.")
        return None, None

text_generator, sentiment_analyzer = load_models()

# Check if models loaded successfully
if text_generator is None or sentiment_analyzer is None:
    st.stop() # Stop the application if models failed to load


# --- AI Agent Functions ---

def generate_response(user_input, history=None):
    """
    Generates a response using the loaded text generation model.
    """
    # Assuming text_generator is loaded and available globally or via st.cache_resource
    if text_generator is None: # Added check
         logging.error("Text generator model not loaded.") # Added logging
         return "I'm sorry, the conversation model is not available.", history # Added graceful exit


    if history is None:
        history = ""

    # Construct the input prompt for the LLM
    # For GPT-2, we can simply append the user input to the history.
    # GPT-2 does not use a specific EOS token for turns like DialoGPT
    # but we can still use it to mark the end of the input if desired,
    # although it's less critical than for DialoGPT's conversational structure.
    # Let's stick to simple concatenation for GPT-2 unless specific fine-tuning was done.
    full_input = history + "User: " + user_input + "\nBot:" # Simple turn-based format

    try:
        # Adjust max_length and generation parameters for GPT-2
        generated_sequence = text_generator(
            full_input,
            max_length=len(full_input) + 200,  # Increased generated tokens further
            # GPT-2 does not have a dedicated pad_token_id like DialoGPT,
            # but the pipeline handles padding implicitly.
            # We can set the eos_token_id to stop generation after a complete response.
            eos_token_id=text_generator.tokenizer.eos_token_id, # Use EOS token to stop generation
            do_sample=True,
            top_k=70, # Increased top_k
            top_p=0.95,
            temperature=0.9, # Increased temperature for more randomness
            num_return_sequences=1,
            truncation=True # Truncate input if it's too long
            # Add repetition penalty if supported and needed
            # repetition_penalty=1.2 # Uncomment and adjust if needed and supported
        )

        generated_text = generated_sequence[0]['generated_text']

        # Extract the generated text that comes AFTER our input prompt ("\nBot:")
        # Find the index where the generated text starts after the prompt
        response_start_marker = "\nBot:"
        response_start_index = generated_text.find(response_start_marker, len(full_input) - len(response_start_marker)) + len(response_start_marker)
        response = generated_text[response_start_index:].strip()

        # Further truncate response if it contains unintended subsequent turns or long generated text
        # Look for common turn-ending patterns like "User:" or "Bot:"
        turn_end_markers = ["\nUser:", "\nBot:"]
        shortest_end_index = len(response)
        for marker in turn_end_markers:
             marker_index = response.find(marker)
             if marker_index != -1 and marker_index < shortest_end_index:
                 shortest_end_index = marker_index

        response = response[:shortest_end_index].strip()


        # Basic safety check on the generated response
        unsafe_keywords = ["kill", "suicide", "harm yourself", "end my life", "unalive myself"] # Add more keywords as needed
        if any(keyword in response.lower() for keyword in unsafe_keywords):
            logging.warning("Potential unsafe response detected.")
            response = "I'm here to support you. Remember, if you are having thoughts of harming yourself, please reach out to a crisis hotline or mental health professional immediately."
            # In a real application, you might want more sophisticated filtering or moderation

        # Update history for the next turn
        # Append the user input and the generated response to the history in the new format
        updated_history = full_input + response + "\n" # Add the generated response and a newline


        return response, updated_history

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, I encountered an error while generating a response. Could you please rephrase that?", history # Return a graceful error message


def assess_severity(user_input, _sentiment_analyzer):
    """
    Assesses the severity of the user's mental health state based on input text.
    Enhanced to be more sensitive to high severity keywords.
    """
    # Assuming sentiment_analyzer is loaded and available globally or via st.cache_resource
    if _sentiment_analyzer is None: # Added check
        logging.warning("Sentiment analyzer not loaded, cannot assess severity.") # Added logging
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
        sentiment_result = _sentiment_analyzer(user_input)[0]
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
    severity = assess_severity(prompt,sentiment_analyzer) # Pass sentiment_analyzer

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
                # Generate response using the LLM
                response_text, st.session_state.conversation_history = generate_response(
                    prompt,
                    st.session_state.conversation_history
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
