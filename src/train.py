import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pickle

from preprocess import clean_text

# Load dataset
df = pd.read_csv('data/fake_job_postings.csv')

# Combine text
df['text'] = (
    df['title'].fillna('') + " " +
    df['description'].fillna('') + " " +
    df['requirements'].fillna('')
)

# Clean text
df['clean_text'] = df['text'].apply(clean_text)

# Features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

# Labels
y = df['fraudulent']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=10)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open('models/model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))