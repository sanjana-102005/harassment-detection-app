import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Train model automatically
# -----------------------------
DATA_FILE = "harassment_dataset.csv"

@st.cache_resource
def train_and_load():
    data = pd.read_csv(DATA_FILE)

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    return model, vectorizer


model, vectorizer = train_and_load()


# -----------------------------
# Laws mapping (India)
# -----------------------------
laws = {
    "verbal": [
        "IPC 504 - Intentional insult to provoke breach of peace",
        "IPC 509 - Word/gesture intended to insult modesty",
    ],
    "physical": [
        "IPC 323 - Voluntarily causing hurt",
        "IPC 352 - Assault or criminal force",
    ],
    "sexual": [
        "IPC 354A - Sexual harassment",
        "IPC 354 - Assault/criminal force to outrage modesty",
        "IPC 375/376 - Rape (if applicable)",
    ],
    "cyber": [
        "IT Act 2000 - Section 66E (Violation of privacy)",
        "IT Act 2000 - Section 67 (Publishing obscene content)",
        "IPC 507 - Criminal intimidation by anonymous communication",
    ],
    "stalking": [
        "IPC 354D - Stalking",
    ],
    "workplace": [
        "POSH Act 2013 - Sexual harassment at workplace",
        "IPC 354A - Sexual harassment",
    ],
    "threat": [
        "IPC 503 - Criminal intimidation",
        "IPC 506 - Punishment for criminal intimidation",
    ],
    "non-harassment": [
        "This may not be harassment based on the input.",
        "If you still feel unsafe, contact a trusted person or police."
    ]
}

# Emergency numbers (India)
emergency_numbers = {
    "Police Emergency": "112",
    "Women Helpline": "1091",
    "Cyber Crime Helpline": "1930",
    "Ambulance": "108",
    "Child Helpline": "1098"
}


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Harassment Detection & Legal Help",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Harassment Detection & Legal Guidance App")
st.write("This app supports **all genders** and helps identify harassment type + possible legal action.")

st.markdown("---")

incident = st.text_area("✍️ Describe the incident:", height=150)
age = st.number_input("Victim Age (optional):", min_value=1, max_value=100, value=18)

if st.button("🔍 Analyze Incident"):
    if incident.strip() == "":
        st.warning("Please enter the incident description.")
    else:
        vec = vectorizer.transform([incident])
        prediction = model.predict(vec)[0]

        st.success(f"✅ Predicted Harassment Type: **{prediction.upper()}**")

        st.subheader("📌 Why this may be harassment?")
        st.write("This prediction is based on patterns learned from the dataset.")

        st.subheader("⚖️ Possible Legal Actions (India)")
        for law in laws.get(prediction, ["No law info available."]):
            st.write("✔️", law)

        if age < 18:
            st.warning("⚠️ Victim is a minor. Child protection laws may apply (POCSO Act, Child Helpline 1098).")

st.markdown("---")

st.subheader("🚨 SOS & Emergency Numbers")
for name, number in emergency_numbers.items():
    st.info(f"📞 {name}: **{number}**")

st.caption("⚠️ Disclaimer: This tool provides guidance only and is not a substitute for legal advice.")
