
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
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database setup
Base = declarative_base()

class Intent(Base):
    __tablename__ = 'intents'
    intent_id = Column(Integer, primary_key=True)
    intent_name = Column(String)

class Category(Base):
    __tablename__ = 'categories'
    category_id = Column(Integer, primary_key=True)
    intent_id = Column(Integer, ForeignKey('intents.intent_id'))
    category_name = Column(String)

class Question(Base):
    __tablename__ = 'questions'
    question_id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey('categories.category_id'))
    question_text = Column(String)

# Set up the database connection
engine = create_engine('sqlite:///database.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def get_categories(intent_id, session):
    return session.query(Category).filter_by(intent_id=intent_id).all()

def get_questions(category_id, session):
    return session.query(Question).filter_by(category_id=category_id).all()

def populate_sample_data(session):
    if session.query(Intent).count() == 0:
        intent1 = Intent(intent_name='greeting')
        intent2 = Intent(intent_name='product inquiry')
        session.add_all([intent1, intent2])
        session.commit()

        category1 = Category(intent_id=intent1.intent_id, category_name='Greetings')
        category2 = Category(intent_id=intent2.intent_id, category_name='Product Inquiry')
        session.add_all([category1, category2])
        session.commit()

        greeting_question1 = Question(category_id=category1.category_id, question_text='Hello! How can I assist you today?')
        product_question1 = Question(category_id=category2.category_id, question_text='What products are you interested in?')

        session.add_all([greeting_question1, product_question1])
        session.commit()
        print("Sample data populated successfully.")
    else:
        print("Sample data already exists. Skipping population.")

def query_based_on_intent(intent_name):
    session = Session()
    intent = session.query(Intent).filter_by(intent_name=intent_name).first()
    if intent:
        categories = get_categories(intent.intent_id, session)
        for category in categories:
            questions = get_questions(category.category_id, session)
            print(f"Category: {category.category_name}")
            for question in questions:
                print(f"- Question: {question.question_text}")
    else:
        print(f"No intent found for: {intent_name}")
    session.close()
intent_to_label = {
    "greeting": 0,
    "product inquiry": 1,
}
# Load the pre-trained LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(intent_to_label))

# Define your intents and their corresponding labels
intent_to_label = {
    "greeting": 0,
    "product inquiry": 1,
}

label_to_intent = {v: k for k, v in intent_to_label.items()}

def classify_intent(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).item()
    return label_to_intent[predictions]

if __name__ == "__main__":
    session = Session()
    populate_sample_data(session)
    session.close()

    # Test the classification
    user_input = "Hi, I want to check the status of my order."
    print("=========")
    predicted_intent = classify_intent(user_input)
    print(f"Predicted intent: {predicted_intent}")

    test_inputs = [
        "Hello, how can I help you?",
        "What products are you interested in?"
    ]

    for input_text in test_inputs:
        predicted_intent = classify_intent(input_text)
        print(f"Input: {input_text} | Predicted Intent: {predicted_intent}")
        query_based_on_intent(predicted_intent)
"""
