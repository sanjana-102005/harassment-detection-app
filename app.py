import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# -----------------------------
# SETTINGS
# -----------------------------
DATA_FILE = "harassment_dataset.csv"

CONFIDENCE_STRONG = 0.65
CONFIDENCE_UNCLEAR = 0.55


# -----------------------------
# Train model automatically
# -----------------------------
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

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, y_train)

    return model, vectorizer


model, vectorizer = train_and_load()


# -----------------------------
# Laws mapping (India) - Basic Guidance
# -----------------------------
laws = {
    "verbal": [
        "IPC 504 - Intentional insult to provoke breach of peace",
        "IPC 506 - Criminal intimidation (if threats included)",
    ],
    "physical": [
        "IPC 323 - Voluntarily causing hurt",
        "IPC 352 - Assault or criminal force",
    ],
    "sexual": [
        "IPC 354A - Sexual harassment",
        "IPC 354 - Assault/criminal force with sexual intent (gender-specific in many cases)",
        "POCSO Act - If victim is a minor",
    ],
    "cyber": [
        "IT Act 2000 - Section 66E (Violation of privacy)",
        "IT Act 2000 - Section 67 (Publishing obscene content)",
        "IPC 507 - Criminal intimidation by anonymous communication",
    ],
    "stalking": [
        "IPC 354D - Stalking (gender-specific in many cases)",
        "IPC 503/506 - Criminal intimidation (if threats included)",
    ],
    "workplace": [
        "POSH Act 2013 - Workplace sexual harassment (for women employees)",
        "Company Internal Complaints Committee (ICC) complaint",
    ],
    "threat": [
        "IPC 503 - Criminal intimidation",
        "IPC 506 - Punishment for criminal intimidation",
    ],
    "non-harassment": [
        "This may not be harassment based on the input.",
        "If you still feel unsafe, talk to a trusted person or contact emergency services."
    ]
}


# -----------------------------
# Emergency numbers (India)
# -----------------------------
emergency_numbers = {
    "Police Emergency": "112",
    "Women Helpline": "1091",
    "Cyber Crime Helpline": "1930",
    "Ambulance": "108",
    "Child Helpline": "1098"
}


# -----------------------------
# KEYWORD SYSTEM (IMPORTANT)
# -----------------------------
CYBER_WORDS = [
    "instagram", "whatsapp", "telegram", "snapchat", "facebook", "gmail", "email",
    "online", "dm", "message", "chat", "account", "password", "hacked", "hack",
    "fake account", "impersonat", "leak", "uploaded", "posted", "shared my photos",
    "nude", "video", "call recording"
]

STALKING_WORDS = [
    "staring", "follow", "following", "waiting outside", "waits outside", "outside my house",
    "outside my hostel", "outside my college", "keeps coming", "tracking my location",
    "unknown numbers", "keeps calling", "watching me", "near my home"
]

SEXUAL_WORDS = [
    "touched", "touching", "groped", "kiss", "forced", "sex", "sexual",
    "body", "nude", "porn", "private parts"
]

PHYSICAL_WORDS = [
    "slap", "slapped", "hit", "punched", "kick", "kicked", "pushed", "grabbed",
    "pulled", "hurt", "beat"
]

THREAT_WORDS = [
    "threat", "threatened", "kill", "ruin", "beat", "harm", "blackmail",
    "leak", "upload", "kidnap"
]

# This list is used to reduce false positives:
# If none of these are present, we treat as "likely non-harassment"
HARMFUL_INTENT_WORDS = list(set(
    CYBER_WORDS + STALKING_WORDS + SEXUAL_WORDS + PHYSICAL_WORDS + THREAT_WORDS +
    ["abuse", "abusive", "harass", "harassment", "bully", "bullying"]
))


# -----------------------------
# PREDICTION FUNCTION (HYBRID)
# -----------------------------
def predict_incident(incident: str):
    text_lower = incident.lower()

    # 1) Rule-based (strong override)
    if any(w in text_lower for w in STALKING_WORDS):
        return "stalking", 0.99, "rule"

    if any(w in text_lower for w in SEXUAL_WORDS):
        return "sexual", 0.99, "rule"

    if any(w in text_lower for w in PHYSICAL_WORDS):
        return "physical", 0.99, "rule"

    if any(w in text_lower for w in THREAT_WORDS):
        return "threat", 0.99, "rule"

    # Cyber should trigger only if cyber words exist
    if any(w in text_lower for w in CYBER_WORDS):
        # But if it is only "shared a reel" or "sent a reel", it is not harassment
        if "reel" in text_lower or "meme" in text_lower:
            if not any(w in text_lower for w in THREAT_WORDS + SEXUAL_WORDS):
                return "non-harassment", 0.90, "rule"
        return "cyber", 0.95, "rule"

    # 2) If no harmful intent words at all → non-harassment (reduces false positives)
    has_harm = any(w in text_lower for w in HARMFUL_INTENT_WORDS)
    if not has_harm:
        return "non-harassment", 0.80, "rule"

    # 3) ML prediction
    vec = vectorizer.transform([incident])
    pred = model.predict(vec)[0]

    proba = model.predict_proba(vec)[0]
    confidence = float(max(proba))

    return pred, confidence, "ml"


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="Harassment Detection & Legal Help",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Harassment Detection & Legal Guidance App")
st.write("This app supports **all genders** and helps identify harassment type + possible legal actions.")

st.markdown("---")

incident = st.text_area("✍️ Describe the incident:", height=150)
age = st.number_input("Victim Age (optional):", min_value=1, max_value=100, value=18)

if st.button("🔍 Analyze Incident"):
    if incident.strip() == "":
        st.warning("Please enter the incident description.")
    else:
        prediction, confidence, method = predict_incident(incident)

        st.success(f"✅ Predicted Type: **{prediction.upper()}**")
        st.info(f"📊 Confidence: **{confidence*100:.2f}%**  |  Method: **{method.upper()}**")

        # -----------------------------
        # Non-harassment response
        # -----------------------------
        if prediction == "non-harassment":
            st.subheader("🌿 Support Message")
            st.write("This incident may not be harassment based on the input.")
            st.write("If you still feel uncomfortable, your feelings are valid.")
            st.write("If it happens repeatedly or becomes unsafe, please seek help.")

        # -----------------------------
        # Unclear response
        # -----------------------------
        elif confidence < CONFIDENCE_UNCLEAR and method == "ml":
            st.subheader("🤔 Not Clear / Needs More Details")
            st.write("This looks like a sensitive situation, but the model is not fully sure.")
            st.write("Try rewriting the incident with more details like:")
            st.write("✔️ Who did it?")
            st.write("✔️ What exactly happened?")
            st.write("✔️ Was it repeated?")
            st.write("✔️ Was there any threat / touching / force / online harassment?")

        # -----------------------------
        # Serious response (Legal)
        # -----------------------------
        else:
            st.subheader("📌 Why this may be harassment?")
            st.write("The system detected harmful intent or repeated unsafe behavior.")

            st.subheader("⚖️ Possible Legal Guidance (India)")
            for law in laws.get(prediction, ["No law info available."]):
                st.write("✔️", law)

            st.subheader("🧾 Evidence to Collect")
            st.write("✔️ Screenshots / chats / emails")
            st.write("✔️ Call logs")
            st.write("✔️ Witness names")
            st.write("✔️ Date, time, and location notes")
            st.write("✔️ Any medical report (if physical harm)")

            if age < 18:
                st.warning("⚠️ Victim is a minor. Child protection laws may apply (POCSO Act, Child Helpline 1098).")

st.markdown("---")

st.subheader("🚨 SOS & Emergency Numbers (India)")
for name, number in emergency_numbers.items():
    st.info(f"📞 {name}: **{number}**")

st.caption("⚠️ Disclaimer: This tool provides guidance only and is not a substitute for professional legal advice.")
