# Email Spam Classifier

A machine learning-powered email spam classifier built with scikit-learn and deployed using Streamlit.

## Features

- **Text Classification**: Uses TF-IDF vectorization and machine learning to classify emails as spam or ham
- **Interactive Web Interface**: Clean Streamlit interface for testing emails
- **Model Pipeline**: Complete ML pipeline with preprocessing and classification
- **Performance Metrics**: Detailed evaluation with confusion matrix and classification reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/email-spam-classifier.git
cd email-spam-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python src/train_model.py
```

4. Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## Dependencies

- streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

## Usage

1. Enter an email message in the text area
2. Click "Classify" to see if it's spam or ham
3. View confidence scores and probabilities

## Model Performance

The model achieves ~96% accuracy on the test set with high precision for spam detection.
