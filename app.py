from flask import Flask, render_template, request
import joblib
import re
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords 
import logging

logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Ensure required NLTK resources are downloaded
nltk.download('stopwords')

# Text cleaning function
def clean(text):
    stemmer = SnowballStemmer("english")
    stopwords_set = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    
    # Remove newline characters
    text = re.sub('\n', '', text)
    
    # Remove words containing digits
    text = re.sub('\w*\d\w*', '', text)
    
    # Tokenize and remove stopwords
    text = [word for word in text.split(' ') if word not in stopwords_set]
    
    # Join cleaned tokens
    text = " ".join(text)
    
    # Apply stemming
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    
    return text

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get user input from the form
#         review = request.form['review']
#         print(f"Raw review: {review}", flush=True)
#         # Clean and preprocess input
#         # Convert the input into a DataFrame
#         new_data = pd.DataFrame([review], columns=['review'])
#         new_data['review'] = new_data['review'].apply(clean)
#         print(f"Cleaned review: {new_data['review'][0]}", flush=True)
        
        
        
#         # Vectorize the input
#         vectorized_review = vectorizer.transform(new_data['review'])
        
#         # Make a prediction
#         prediction = model.predict(vectorized_review)
#         print(f"Prediction: {prediction}", flush=True)
        
#         # Convert prediction to a human-readable label
#         sentiment = "Positive" if prediction[0] == 'Positive' else "Negative"

        
#         # Return the result to the user
#         return render_template('index.html', prediction_text=f"Sentiment: {sentiment}", review=review)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form.get('Review')
        cleaned_review = clean(review)
        vectorized_review = vectorizer.transform([cleaned_review])
        prediction = model.predict(vectorized_review)

        result = "Positive" if prediction[0] == 'Positive' else "Negative"
        print(f"Prediction: {result}")  # Debugging: check the result in the console
        return render_template('index.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, render_template, request
# import joblib
# import re
# import pandas as pd
# import nltk
# from nltk.stem import SnowballStemmer
# from nltk.corpus import stopwords
# import logging

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # Initialize Flask app
# app = Flask(__name__)

# # Load the saved model and vectorizer
# model = joblib.load('model.pkl')
# vectorizer = joblib.load('vectorizer.pkl')

# # Ensure required NLTK resources are downloaded
# nltk.download('stopwords')

# # Text cleaning function
# def clean(text):
#     stemmer = SnowballStemmer("english")
#     stopwords_set = set(stopwords.words('english'))

#     # Convert to lowercase
#     text = str(text).lower()

#     # Remove HTML tags
#     text = re.sub('<.*?>+', '', text)

#     # Remove newline characters
#     text = re.sub('\n', '', text)

#     # Remove words containing digits
#     text = re.sub('\w*\d\w*', '', text)

#     # Tokenize and remove stopwords
#     text = [word for word in text.split(' ') if word not in stopwords_set]

#     # Join cleaned tokens
#     text = " ".join(text)

#     # Apply stemming
#     text = [stemmer.stem(word) for word in text.split(' ')]
#     text = " ".join(text)

#     return text

# # Home route
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get user input from the form
#         review = request.form.get('review', '')
#         logging.debug(f"Raw review: {review}")

#         # Clean and preprocess input
#         new_data = pd.DataFrame([review], columns=['review'])
#         new_data['review'] = new_data['review'].apply(clean)
#         logging.debug(f"Cleaned review: {new_data['review'][0]}")

#         # Vectorize the input
#         vectorized_review = vectorizer.transform(new_data['review'])

#         # Make a prediction
#         prediction = model.predict(vectorized_review)
#         logging.debug(f"Prediction: {prediction}")

#         # Convert prediction to a human-readable label
#         sentiment = "Positive" if prediction[0] == 'Positive' else "Negative"

#         # Return the result to the user
#         return render_template('index.html', prediction_text=f"Sentiment: {sentiment}", review=review)

# if __name__ == '__main__':
#     app.run(debug=True)
