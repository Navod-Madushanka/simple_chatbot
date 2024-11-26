import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import random

# Step 1: Load and preprocess the text file
file_path = "new_data.csv"  # Replace with your file path
dialogues = []

# Read and preprocess the dialogue file
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:  # Ignore empty lines
            try:
                speaker, message = line.split(":", 1)  # Split into speaker and message
                dialogues.append({"Speaker": speaker.strip(), "Message": message.strip()})
            except ValueError:
                print(f"Skipping malformed line: {line}")

# Step 2: Convert the dialogue list to a DataFrame
data = pd.DataFrame(dialogues)

# Step 3: Generate dummy sentiment labels (e.g., Positive, Negative, Neutral)
# Modify this logic if you have specific labeling criteria.
data["Sentiment"] = data["Message"].apply(
    lambda x: "Positive" if "good" in x.lower() or "great" in x.lower()
    else "Negative" if "bad" in x.lower() or "sad" in x.lower()
    else "Neutral"
)

# Check the preprocessed data
print("Sample of processed data:")
print(data.head())

# Step 4: Select the relevant columns
X = data["Message"]  # The input text
y = data["Sentiment"]  # The sentiment label

# Handle missing values in X
X = X.fillna("")

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 7: Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Step 8: Evaluate the model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Step 9: Function to get a response based on sentiment
def get_response(user_input):
    user_input = user_input.lower()  # Preprocess the input
    try:
        # Transform the user input to match the trained model's expectations
        user_input_vector = vectorizer.transform([user_input])
        predicted_label = model.predict(user_input_vector)[0]  # Predict the label

        # Generate responses based on the predicted label
        if predicted_label == "Neutral":
            return random.choice([
                "Interesting! Can you tell me more?",
                "Thanks for sharing. What else is on your mind?",
                "Hmm, that's something to think about."
            ])
        else:
            return random.choice([
                f"Got it! You said: {user_input}. Can we expand on that?",
                f"That's intriguing! What would you like to discuss further?",
                "Thank you for sharing! What's next?"
            ])
    except Exception as e:
        print(f"Error: {e}")
        return "I'm sorry, I couldn't process that. Could you rephrase?"

# Step 10: Main chat loop
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
