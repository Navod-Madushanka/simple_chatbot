import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Step 1: Load the CSV file
data = pd.read_csv("new_data.csv", header=None)  # Replace with your actual path
data.columns = ["ID", "Game", "Sentiment", "Message"]  # Assign column names based on the structure

# Step 2: Select the relevant columns
X = data["Message"]  # The message text
y = data["Sentiment"]  # The sentiment label

# Handle missing values in X by replacing NaN with an empty string
X = X.fillna("")

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# Step 5: Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Function to get a response based on sentiment (since "intent" labels like greeting aren't in the data)
def get_response(user_input):
    # Preprocess the input (e.g., lowercase it)
    user_input = user_input.lower()

    try:
        user_input_vector = vectorizer.transform([user_input])  # Transform input
        sentiment = model.predict(user_input_vector)[0]  # Predict sentiment

        # Respond based on the identified sentiment
        if sentiment == "Positive":
            return random.choice(["That's great to hear! How can I assist you today?",
                                  "I'm glad you're feeling positive!",
                                  "It sounds like things are going well!"])
        elif sentiment == "Negative":
            return random.choice(["I'm sorry to hear that. I'm here if you need to talk.",
                                  "It seems you're feeling down. Let me know how I can help.",
                                  "I’m here for you. Feel free to share what's on your mind."])
        else:
            return "I'm not sure I understand. Could you rephrase?"
    except Exception as e:
        print(f"Error: {e}")
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
