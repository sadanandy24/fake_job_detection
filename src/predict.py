import pickle
from preprocess import clean_text

# Load model and vectorizer
model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

def predict_job(text):
    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "⚠️ Fake Job Posting"
    else:
        return "✅ Real Job Posting"


# Test input
if __name__ == "__main__":
    sample = input("Enter job description: ")
    result = predict_job(sample)
    print(result)