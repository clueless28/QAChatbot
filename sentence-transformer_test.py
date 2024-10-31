from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
intents = {
    "greeting": "Hello, hi, hey, how are you?",
    "product inquiry": "I want to buy a product, I need information on a product",
    "feedback": "I have some feedback, I want to give a review",
    "customer support": "I need help, I have an issue, support needed",
    "order status": "Where is my order, track my order, order details",
    "appointment scheduling": "I want to schedule an appointment",
    "information request": "I need information, can you tell me more?",
    "closing statement": "Thank you, goodbye, see you later",
    "account management": "Help with my account, account details",
}


intent_embeddings = {intent: model.encode(description) for intent, description in intents.items()}
follow_up_questions = {
    "greeting": "How can I assist you today?",
    "product inquiry": "Can you please specify the product you are inquiring about?",
    "feedback": "What kind of feedback do you have?",
    "customer support": "What would you like help with in your account?",
    "order status": "Can you provide your order number?",
    "appointment scheduling": "Do you want to schedule an appointment?",
    "information request": "Is there anything else you would like to know?",
    "closing statement": "Can I help you with anything else?",
    "account management": "What would you like help with in your account?",
}

def classify_intent(input_text):
    user_embedding = model.encode(input_text)
    similarities = {
        intent: util.pytorch_cos_sim(user_embedding, embedding).item()
        for intent, embedding in intent_embeddings.items()
    }
    

    predicted_intent = max(similarities, key=similarities.get)
    return predicted_intent

def get_follow_up_question(predicted_intent):
    return follow_up_questions.get(predicted_intent, "How can I assist you further?")


if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Thank you for chatting! Goodbye!")
            break
        predicted_intent = classify_intent(user_input)
        print(f"Predicted intent: {predicted_intent}")
        follow_up_question = get_follow_up_question(predicted_intent)
        print(f"Follow-up question: {follow_up_question}")
