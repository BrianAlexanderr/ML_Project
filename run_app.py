import streamlit as st
import joblib
import PyPDF2
import docx
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

with open('spam_classifier.pkl', 'rb') as f:
    model = joblib.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = joblib.load(f)

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = ' '.join([lemma.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def main():
    st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

    # Header
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📧 Spam Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #646464;'>Enter text or upload a document and find out if it's spam!</p>", unsafe_allow_html=True)
    st.write("---")

    input_text = ""
    input_choice = st.radio("Choose input method:", ["Text Input", "Upload File"], horizontal=True)

    if input_choice == "Text Input":
        input_text = st.text_area("Enter your message here:", height=150)
    else:
        uploaded_file = st.file_uploader("Upload a file (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                input_text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                input_text = read_pdf(uploaded_file)
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                input_text = read_docx(uploaded_file)
            else:
                st.error("Unsupported file type.")

    if st.button("Classify", type="primary"):
        if input_text.strip() == "":
            st.warning("Please enter or upload some text.")
        else:
            X = preprocess(input_text)
            X = vectorizer.transform([X]).toarray()

            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            confidence = max(proba) * 100

            color = "green" if prediction == "ham" else "red"
            emoji = "📮" if prediction == "ham" else "🚫"
            pred_name = 'Not Spam' if prediction == "ham" else "Spam"
            with st.container():
                st.markdown(
                    f"<h2 style='color:{color}; font-weight:bold;'>{emoji} Prediction: {pred_name}</h2>", unsafe_allow_html=True
                )
                st.markdown(f"**Confidence:** {confidence:.2f} %")
                with st.expander("Show Input Text"):
                    st.write(input_text)

    st.write("---")


if __name__ == "__main__":
    main()

