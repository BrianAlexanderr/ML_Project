# ML_Project: Spam vs. Ham Message Classification

Welcome to **ML_Project**! This repository tackles the classic problem of predicting whether a message is spam or ham (not spam) using advanced ensemble machine learning techniques, and delivers an interactive experience via Streamlit.

## 🚀 Project Overview

- **Objective:** Accurately classify messages as spam or ham using an ensemble of machine learning models.
- **Solution:** We leverage the power of multiple algorithms, combining their strengths for superior predictive accuracy.
- **Deployment:** The model is deployed with [Streamlit](https://streamlit.io/), providing an intuitive web interface for real-time message classification.

## ✨ Features

- Preprocessing and feature extraction for text data
- Multiple ensemble models (e.g., Random Forest, Gradient Boosting, Voting Classifier)
- Interactive Streamlit app for message prediction
- Easy-to-use UI: Paste your message and get instant results!
- Visualizations of model performance

## 🛠️ Tech Stack

- **Python**
- **scikit-learn** (ensemble models)
- **pandas**, **numpy** (data manipulation)
- **Streamlit** (web app deployment)
- **NLTK / spaCy** (text processing, if applicable)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/BrianAlexanderr/ML_Project.git
   cd ML_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

## 🧑‍💻 Usage

- Launch the Streamlit app.
- Enter a message in the input box.
- The app will classify the message as **Spam** or **Ham** and show the prediction confidence.

## 📝 Example

```text
Input: "Congratulations! You've won a free ticket. Call now to claim."
Output: Spam (Confidence: 98%)
```

## 📊 Results

- Achieved high accuracy and robustness using the ensemble approach.
- Detailed evaluation metrics and visualizations are available in the app.

## 🤝 Contributing

Contributions are welcome! Please fork the repo, create a pull request, or open an issue for suggestions and improvements.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙋‍♂️ Author

Developed by [BrianAlexanderr](https://github.com/BrianAlexanderr).

