
import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from db_setup import query_based_on_intent
# Load the pre-trained LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"  # Ensure you are using the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as the pad token

# Define your intents and their corresponding labels
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

def classify_intent(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Perform inference
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)

    # Get the predicted intent
    predictions = outputs.logits.argmax(dim=-1).item()
    return label_to_intent[predictions]

# Example usage
if __name__ == "__main__":
    user_input = "Hi, I want to check the status of my order."
    predicted_intent = classify_intent(user_input)
    print(f"Predicted intent: {predicted_intent}")

    # Test with different user inputs
    test_inputs = [
        "Hello, how can I help you?",
        "What is the price of the new smartphone?",
        "I want to give some feedback about my last purchase.",
        "Can you assist me with my account?",
        "When will my order arrive?"
    ]

    for input_text in test_inputs:
        predicted_intent = classify_intent(input_text)
        print(f"Input: {input_text} | Predicted Intent: {predicted_intent}")
        query_based_on_intent(predicted_intent)
