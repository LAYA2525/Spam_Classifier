"""
Streamlit front-end.

Run:
    streamlit run app/streamlit_app.py
"""
from pathlib import Path
import sys
import streamlit as st

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from utils import load_artifact

st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
# Use the complete pipeline instead of separate components
try:
    pipeline = load_artifact(MODEL_DIR / "spam_pipeline.pkl")
except FileNotFoundError:
    # Fallback to separate components if pipeline doesn't exist
    model = load_artifact(MODEL_DIR / "spam_classifier.pkl")
    vectorizer = load_artifact(MODEL_DIR / "vectorizer.pkl")
    pipeline = None

st.title(" Email Spam Classifier")
st.markdown(
    "Enter an email (or SMS) message below and click **Classify** "
    "to see whether the system thinks it's **Spam** or **Ham**."
)

user_msg = st.text_area("Your message", height=200)

if st.button("Classify"):
    if user_msg.strip() == "":
        st.warning("Please enter a message first.")
    else:
        if pipeline is not None:
            # Use the complete pipeline
            pred = pipeline.predict([user_msg])[0]
            # Get prediction probability for confidence
            proba = pipeline.predict_proba([user_msg])[0]
            confidence = max(proba) * 100
        else:
            # Fallback to separate components
            msg_vec = vectorizer.transform([user_msg])
            pred = model.predict(msg_vec)[0]
            proba = model.predict_proba(msg_vec)[0]
            confidence = max(proba) * 100
        
        if pred == 1:
            st.error(f"🚫 **SPAM!** (Confidence: {confidence:.1f}%)")
        else:
            st.success(f"✅ **Ham** (Confidence: {confidence:.1f}%)")
        
        # Show prediction probabilities
        st.write(f"Ham probability: {proba[0]:.3f}")
        st.write(f"Spam probability: {proba[1]:.3f}")
