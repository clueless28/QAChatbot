import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

model_name = "meta-llama/Llama-3.2-3B"  # Model for classification
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(model_name)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

intent_to_label = {
    "greeting": 0,
    "product inquiry": 1,
    "feedback": 2,
    "customer support": 3,
    "order status": 4,
    "appointment scheduling": 5,
    "information request": 6,
    "closing statement": 7,
    "account management": 8,
}

label_to_intent = {v: k for k, v in intent_to_label.items()}
follow_up_questions = [
    "Can you please specify the product you are inquiring about?",
    "What kind of feedback do you have?",
    "Can you provide your order number?",
    "What would you like help with in your account?",
    "Is there anything else you would like to know?",
    "How can I assist you today?",
    "Do you want to schedule an appointment?",
    "Can I help you with anything else?",
]

def classify_intent(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).item()
    return label_to_intent[predictions]

def get_most_relevant_question(user_input):
    # Encode user input and follow-up questions
    user_embedding = embedding_model.encode(user_input, convert_to_tensor=True)
    questions_embeddings = embedding_model.encode(follow_up_questions, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(user_embedding, questions_embeddings)

    # Get the index of the highest score
    most_relevant_index = torch.argmax(cosine_scores).item()
    return follow_up_questions[most_relevant_index]


if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        
        # Exit condition
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Thank you for chatting! Goodbye!")
            break
        
        # Classify intent
        predicted_intent = classify_intent(user_input)
        print(f"Predicted intent: {predicted_intent}")

        # Get the most relevant follow-up question based on user input
        follow_up_question = get_most_relevant_question(user_input)
        print(f"Follow-up question: {follow_up_question}")
