import streamlit as st
from transformers import pipeline

# Initialize models (assuming these are initialized once)
# This needs to be done only once, perhaps outside this function or with caching
# @st.cache_resource
# def load_models():
#    text_generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")
#    sentiment_analyzer = pipeline("sentiment-analysis")
#    return text_generator, sentiment_analyzer

# text_generator, sentiment_analyzer = load_models()

# Re-define functions here to ensure they are available in the Streamlit context
# In a real application, these would be in separate modules and imported.
# Since we are in a notebook, redefine for clarity.

# Re-define generate_response function with basic safety check
def generate_response(user_input, history=None):
  """Generates a response using the loaded text generation model with a basic safety check."""
  # Assuming text_generator is globally available or passed in
  if 'text_generator' not in st.session_state:
       st.session_state.text_generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")

  text_generator = st.session_state.text_generator

  if history is None:
      history = ""

  full_input = history + user_input + text_generator.tokenizer.eos_token

  generated_sequence = text_generator(
      full_input,
      max_length=len(full_input) + 50, # Generate up to 50 new tokens
      pad_token_id=text_generator.tokenizer.eos_token_id,
      do_sample=True,
      top_k=50,
      top_p=0.95,
      temperature=0.7,
      num_return_sequences=1,
      truncation=True
  )

  generated_text = generated_sequence[0]['generated_text']
  # Find the index of the first occurrence of the EOS token after the input
  eos_index_after_input = generated_text[len(full_input):].find(text_generator.tokenizer.eos_token)
  if eos_index_after_input != -1:
      # If EOS token is found, take the text up to that point
      response = generated_text[len(full_input):len(full_input) + eos_index_after_input].strip()
  else:
      # Otherwise, take the rest of the generated text
      response = generated_text[len(full_input):].strip()

  # Basic safety check on the generated response
  unsafe_keywords = ["kill", "suicide", "harm yourself"] # Add more keywords as needed
  if any(keyword in response.lower() for keyword in unsafe_keywords):
      print("Potential unsafe response detected, providing a standard safe response.")
      response = "I'm here to support you. Remember, if you are having thoughts of harming yourself, please reach out to a crisis hotline or mental health professional immediately."
      # Log this incident for review in a real application

  updated_history = history + user_input + text_generator.tokenizer.eos_token + response + text_generator.tokenizer.eos_token

  return response, updated_history

# Re-define assess_severity function
def assess_severity(user_input):
    """
    Assesses the severity of the user's mental health state based on input text.
    Enhanced to be more sensitive to high severity keywords.
    """
    # Assuming sentiment_analyzer is globally available or passed in
    if 'sentiment_analyzer' not in st.session_state:
         st.session_state.sentiment_analyzer = pipeline("sentiment-analysis")

    sentiment_analyzer = st.session_state.sentiment_analyzer


    # Define keywords indicating higher severity - include variations
    high_severity_keywords = ["suicide", "kill myself", "ending it all", "hopeless", "no point",
                              "end my life", "harm myself", "want to die", "can't go on"]
    medium_severity_keywords = ["sad", "depressed", "anxious", "stressed", "struggling", "down",
                                "overwhelmed", "lonely", "empty"]

    # Check for high severity keywords - immediate flag
    user_input_lower = user_input.lower()
    for keyword in high_severity_keywords:
        if keyword in user_input_lower:
            return 'high'

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

# Re-define provide_recommendations function
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


# Set the title of the Streamlit application
st.title("Mental Health Support AI Agent")

# Add a prominent disclaimer at the beginning
st.warning("""
**Disclaimer:** This AI agent is for informational and conversational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing a mental health crisis or have serious concerns about your well-being, please consult a qualified healthcare provider or contact a crisis hotline immediately.
""")


# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = ""

# Initialize messages list in session state to store chat history for display
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
                # Generate response using the LLM
                response_text, st.session_state.conversation_history = generate_response(prompt, st.session_state.conversation_history)

                # Provide recommendations based on severity (already assessed)
                recommendations = provide_recommendations(severity)

                # Combine response and recommendations
                full_response = response_text + "\n\n" + recommendations

                st.markdown(full_response)

        # Add assistant's full response (including recommendations) to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})