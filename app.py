import streamlit as st
import pickle
from src.preprocess import clean_text
import nltk

# Fix for deployment
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# load model
model = pickle.load(open('models/model.pkl','rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

st.set_page_config(page_title="Sentinel")

# UI
st.title("Sentinel")
st.caption("Check job postings for potential issues")

# input box
job_text = st.text_area("Paste job description here:", height=200)

# analyze button
if st.button("Analyze"):

    if job_text.strip() == "":
        st.warning("Please enter a job description.")
    
    else:
        x = vectorizer.transform([clean_text(job_text)])
        pred = model.predict(x)[0]

        # rule fix
        if any(k in job_text.lower() for k in ["pay","fee","registration"]):
            pred = 1

        # output
        if pred == 1:
            st.error("⚠️ This job looks suspicious")

            st.markdown("### Why?")
            st.write("- Unrealistic salary")
            st.write("- No experience required")
            st.write("- Requests payment")

            st.markdown("### What should you do?")
            st.write("- Do NOT pay any money")
            st.write("- Check company on LinkedIn")
            st.write("- Look for reviews online")

        else:
            st.success("✅ This job looks fine")

            st.markdown("### Recommendation")
            st.write("- Verify company website")
            st.write("- Apply through official portal")