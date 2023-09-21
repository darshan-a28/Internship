import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

tokenizer = RegexpTokenizer(r"\w+")
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def getCleanedText(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    clean_text = " ".join(stemmed_tokens)
    return clean_text

# Training data
X = ["This was really awesome an awesome movie",
     "Great movie! I liked it a lot",
     "Happy Ending! Awesome Acting by hero",
     "loved it!",
     "Bad not up to the mark",
     "Could have been better",
     "really Disappointed by the movie"]
print(X)

y = ["positive", "positive", "positive", "positive", "negative", "negative", "negative"]

# Preprocess the training data
X_train_clean = [getCleanedText(text) for text in X]

# Vectorize the training data
cv = CountVectorizer(ngram_range=(1, 2))
X_train_vec = cv.fit_transform(X_train_clean).toarray()

# Train the model
mn = MultinomialNB()
mn.fit(X_train_vec, y)

# Take user input for text
user_input = input("Enter text: ")

# Preprocess user input
user_input_clean = getCleanedText(user_input)

# Vectorize user input
user_input_vec = cv.transform([user_input_clean]).toarray()

# Predict sentiment for user input
user_sentiment = mn.predict(user_input_vec)[0]

# Print the predicted sentiment
print("Predicted Sentiment:", user_sentiment)

# Test data for model evaluation
X_test = ["it was bad","loved it"]
y_test = ["negative","positive"]

# Preprocess the test data
X_test_clean = [getCleanedText(text) for text in X_test]

# Vectorize the test data
X_test_vec = cv.transform(X_test_clean).toarray()

# Predict sentiment for test data
y_pred = mn.predict(X_test_vec)

# Evaluate the model using accuracy
accuracy = accuracy_score(y_test, y_pred)*100

# Print the accuracy
print("Model Accuracy on Test Data:",accuracy)

