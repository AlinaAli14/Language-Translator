import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample FAQs
faqs = {
    "What are your hours of operation?": "We are open from 9 AM to 5 PM, Monday to Friday.",
    "Where are you located?": "We are located at 123 Main Street, Springfield.",
    "How can I contact support?": "You can contact support by emailing support@example.com or calling 123-456-7890.",
    "What is your return policy?": "Our return policy allows for returns within 30 days of purchase with a receipt.",
    "Do you offer international shipping?": "Yes, we offer international shipping to many countries. Please check our shipping policy for more details."
}

# Initialize the vectorizer
vectorizer = CountVectorizer().fit_transform(faqs.keys())
vectors = vectorizer.toarray()

# Define a function to find the most similar question
def get_response(user_query):
    user_vec = vectorizer.transform([user_query]).toarray()
    cosine_similarities = cosine_similarity(user_vec, vectors).flatten()
    best_match_index = np.argmax(cosine_similarities)
    
    if cosine_similarities[best_match_index] > 0.1:  # Set a threshold for similarity
        return list(faqs.values())[best_match_index]
    else:
        return "I'm sorry, I don't understand your question. Can you please rephrase?"

# Main loop for the chatbot
def chat():
    print("Hello! How can I assist you today? Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = get_response(user_input)
        print("Bot:", response)

# Uncomment the line below to run the chatbot
# chat()