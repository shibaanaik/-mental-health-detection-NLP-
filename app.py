import streamlit as st
import joblib
import sklearn  # ensures sklearn is available for joblib


model = joblib.load("mental_health_model.joblib")
vectorizer = joblib.load("mental_health_vectorizer.joblib")

label_map = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar disorder",
    3: "Personality disorder",
    4: "Anxiety"
}


st.set_page_config(page_title="Mental Health Distress Detector", page_icon="🧠", layout="centered")

st.title("🧠 Mental Health Distress Detection")
st.write("Enter your thoughts, and the AI model will predict the mental health condition category.")

# User input
user_input = st.text_area("Write your feelings or thoughts here:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before analysis.")
    else:
        # Transform input text
        input_features = vectorizer.transform([user_input])

        # Predict class
        prediction = model.predict(input_features)[0]
        label = label_map[prediction]

        # Display prediction
        st.success(f"### 🩺 Predicted Mental Health Condition: **{label}**")

        # Show prediction confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_features)[0]
            st.subheader("Prediction Confidence")
            for i, prob in enumerate(probs):
                st.write(f"{label_map[i]}: {prob:.2%}")

#footer
st.markdown(
    """
    <hr style="margin-top:40px;margin-bottom:10px;">
    <div style="text-align:center; color:gray;">
        Made with ❤️ by <b>Shibaa</b>
    </div>
    """,
    unsafe_allow_html=True
)
