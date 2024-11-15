import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Expanded Training Data
training_data = [
    # Greetings
    ("hello", "greeting"), ("hi", "greeting"), ("hey", "greeting"), ("howdy", "greeting"),
    ("how are you", "greeting"), ("good morning", "greeting"), ("good evening", "greeting"),
    ("greetings", "greeting"), ("what's up", "greeting"), ("yo", "greeting"), ("hi there", "greeting"),
    ("how is it going", "greeting"), ("sup", "greeting"), ("good to see you", "greeting"),
    ("hello there", "greeting"), ("hey there", "greeting"), ("how are you doing", "greeting"),
    ("nice to meet you", "greeting"), ("hey, what's up", "greeting"), ("what's happening", "greeting"),

    # Farewells
    ("bye", "farewell"), ("goodbye", "farewell"), ("see you later", "farewell"), ("take care", "farewell"),
    ("farewell", "farewell"), ("catch you later", "farewell"), ("talk to you soon", "farewell"),
    ("I'm leaving", "farewell"), ("got to go", "farewell"), ("later", "farewell"), ("peace out", "farewell"),
    ("so long", "farewell"), ("bye bye", "farewell"), ("have a good one", "farewell"),
    ("see ya", "farewell"), ("adios", "farewell"), ("I’m signing off", "farewell"),
    ("see you around", "farewell"), ("take it easy", "farewell"), ("good night", "farewell"),

    # Name Queries
    ("what's your name", "name_query"), ("who are you", "name_query"), ("what should I call you", "name_query"),
    ("do you have a name", "name_query"), ("tell me your name", "name_query"), ("introduce yourself", "name_query"),
    ("are you named", "name_query"), ("your name please", "name_query"), ("may I know your name", "name_query"),
    ("name?", "name_query"), ("can you tell me your name", "name_query"), ("identify yourself", "name_query"),
    ("do you go by a name", "name_query"), ("what do they call you", "name_query"),
    ("what’s your identity", "name_query"), ("any name", "name_query"), ("what do you go by", "name_query"),
    ("do you have an identity", "name_query"), ("how can I address you", "name_query"), ("who is this", "name_query"),

    # Joke Requests
    ("tell me a joke", "joke_request"), ("make me laugh", "joke_request"), ("can you tell jokes", "joke_request"),
    ("do you know any jokes", "joke_request"), ("funny joke", "joke_request"), ("make me smile", "joke_request"),
    ("do you have jokes", "joke_request"), ("joke?", "joke_request"), ("laugh?", "joke_request"),
    ("make me giggle", "joke_request"), ("say something funny", "joke_request"), ("humor me", "joke_request"),
    ("I need a laugh", "joke_request"), ("cheer me up with a joke", "joke_request"), ("know any good jokes?", "joke_request"),
    ("what’s funny?", "joke_request"), ("anything funny?", "joke_request"), ("say something humorous", "joke_request"),
    ("a good joke, please", "joke_request"), ("do you know any humor?", "joke_request"),

    # Negative Mood
    ("I feel sad", "mood_negative"), ("I'm not feeling well", "mood_negative"), ("I'm so unhappy", "mood_negative"),
    ("life is tough", "mood_negative"), ("I feel terrible", "mood_negative"), ("I'm feeling down", "mood_negative"),
    ("it's a bad day", "mood_negative"), ("nothing's going right", "mood_negative"), ("I'm upset", "mood_negative"),
    ("I need help", "mood_negative"), ("I'm feeling depressed", "mood_negative"), ("everything is bad", "mood_negative"),
    ("I feel lonely", "mood_negative"), ("life sucks", "mood_negative"), ("I feel miserable", "mood_negative"),
    ("I'm in pain", "mood_negative"), ("this is the worst", "mood_negative"), ("I feel broken", "mood_negative"),
    ("things are hard", "mood_negative"), ("I feel defeated", "mood_negative"),

    # Positive Mood
    ("I'm very happy today", "mood_positive"), ("I feel great", "mood_positive"), ("life is amazing", "mood_positive"),
    ("I'm so excited", "mood_positive"), ("today is a wonderful day", "mood_positive"), ("everything is awesome", "mood_positive"),
    ("I'm on top of the world", "mood_positive"), ("I'm thrilled", "mood_positive"), ("I love life", "mood_positive"),
    ("this is the best day", "mood_positive"), ("I feel fantastic", "mood_positive"), ("everything is perfect", "mood_positive"),
    ("this is wonderful", "mood_positive"), ("I'm in a great mood", "mood_positive"), ("I'm full of joy", "mood_positive"),
    ("things are going well", "mood_positive"), ("life is great", "mood_positive"), ("I'm super happy", "mood_positive"),
    ("I'm smiling all day", "mood_positive"), ("this day is amazing", "mood_positive"),
]


# Prepare the data
X, y = zip(*training_data)  # Separate inputs and labels
vectorizer = CountVectorizer()  # Initialize vectorizer
X = vectorizer.fit_transform(X)  # Transform text to numeric vectors

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Function to get a response based on intent
def get_response(user_input):
    # Preprocess the input (e.g., lowercase it)
    user_input = user_input.lower()

    try:
        user_input_vector = vectorizer.transform([user_input])  # Transform input
        intent = model.predict(user_input_vector)[0]  # Predict intent

        # Respond based on the identified intent
        if intent == "greeting":
            return random.choice(["Hello!", "Hi there!", "Hey! How can I help you?"])
        elif intent == "farewell":
            return random.choice(["Goodbye! Take care.", "See you later!", "It was nice talking with you."])
        elif intent == "name_query":
            return "I'm an AI chatbot without a name."
        elif intent == "joke_request":
            return "Why don’t scientists trust atoms? Because they make up everything!"
        elif intent == "mood_negative":
            return "I'm sorry to hear that. I'm here if you need to talk."
        elif intent == "mood_positive":
            return "That's great to hear! How can I assist you today?"
        else:
            return "I'm not sure I understand. Could you rephrase?"
    except:
        return "I'm sorry, I couldn't process that. Could you rephrase?"

# Main chat loop
def start_chat():
    print("Hello! I'm your AI chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", get_response(user_input))

# Run the chatbot
if __name__ == "__main__":
    start_chat()
