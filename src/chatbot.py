import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import random
from nltk.corpus import stopwords

# Download nltk packages
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Load the saved model and other required data
model = load_model('D:\Jupyter\EDC\chatbot\Makoto_EDC\src\chatbot_model_test.h5')

with open('D:\Jupyter\EDC\chatbot\Makoto_EDC\src\json\words.json') as file:
    words_data = json.load(file)
    words = words_data['words']

with open('D:\Jupyter\EDC\chatbot\Makoto_EDC\src\json\classes.json') as file:
    classes_data = json.load(file)
    classes = classes_data['classes']

with open('D:\Jupyter\EDC\chatbot\Makoto_EDC\src\json\intents.json') as file:
    intents_data = json.load(file)
    intents_list = intents_data['intents']

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

# Function to preprocess the user's input
def preprocess_input(sentence):
    word_list = nltk.word_tokenize(sentence)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
    bag = [1 if word in word_list else 0 for word in words]
    return np.array(bag)

# Function to predict the intents and their confidences
def get_response(input_text):
    bag = preprocess_input(input_text)
    prediction = model.predict(np.array([bag]))[0]
    predicted_intent_indices = np.argsort(prediction)[::-1][:2]  # Get the indices of top two predicted intents
    predicted_intents = [classes[index] for index in predicted_intent_indices]
    confidences = [prediction[index] for index in predicted_intent_indices]
    return predicted_intents, confidences

# Function to get the appropriate response based on the predicted intent
def get_selected_response(intents):
    for intent in intents:
        for intent_data in intents_list:
            if intent_data['tag'] == intent:
                responses = intent_data['responses']
                return np.random.choice(responses)

# Main loop to interact with the chatbot
print("Bot: Hi! How can I assist you? (type 'quit' to exit)")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'quit':
        print("Bot: Goodbye!")
        break
    
    predicted_intents, confidences = get_response(user_input)
    
    print("Predicted Intents:")
    for intent, confidence in zip(predicted_intents, confidences):
        print(f"{intent}: {confidence:.2f}")
    
    response = get_selected_response(predicted_intents)
    
    if response is not None:
        print("Bot:", response)
    else:
        print("Bot: I'm sorry, but I didn't understand your question.")