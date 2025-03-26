import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit App UI
st.title("📧 Phishing Email Detector")
st.write("Enter an email message below to check if it's spam or phishing.")

email_input = st.text_area("✍️ Paste the email content here:")

if st.button("🔍 Detect Phishing"):
    if email_input.strip():
        email_vectorized = vectorizer.transform([email_input])  # Transform input text
        prediction = model.predict(email_vectorized.toarray())[0] # Predict
        result = "🚨 **Phishing/Spam Detected!**" if prediction == 1 else "✅ **Safe Email**"
        st.subheader(result)
    else:
        st.warning("⚠️ Please enter an email to analyze.")