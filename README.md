# Fake Job Detection Project

## Description
This project is a machine learning application built with Streamlit that analyzes job postings to detect potentially fraudulent or suspicious job listings. It uses natural language processing and a trained model to classify job descriptions.

## Features
- Text preprocessing and cleaning
- Machine learning model for prediction
- Streamlit web interface for easy analysis
- Rule-based enhancements for better detection

## Installation
1. Clone or download the project.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Ensure the model and vectorizer files are in the `models/` directory.
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Open the provided URL in your browser.
4. Paste a job description into the text area and click "Analyze" to check for suspicious content.

## Project Structure
- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `data/`: Contains the dataset (fake_job_postings.csv)
- `models/`: Trained model and vectorizer files
- `src/`: Source code modules
  - `preprocess.py`: Text cleaning functions
  - `predict.py`: Prediction logic
  - `train.py`: Model training script

## Dependencies
- streamlit
- pandas
- scikit-learn
- xgboost
- nltk

## Model Training
To retrain the model, use the scripts in the `src/` directory with the data in `data/fake_job_postings.csv`.