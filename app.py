
import streamlit as st
import os
import tempfile
from modules.stt import speech_to_text
from modules.voice import text_to_speech

st.set_page_config(
    page_title="Annadata AI",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Annadata AI")
st.subheader("Your Voice Assistant for Farming")
st.write(
    "Ask your farming questions in your preferred language."
)


language_options = {
    "Assamese (অসমীয়া)": "as-IN",
    "Bengali (বাংলা)": "bn-IN",
    "Bodo (बड़ो)": "brx-IN",
    "Dogri (डोगरी)": "doi-IN",
    "English (India)": "en-IN",
    "Gujarati (ગુજરાતી)": "gu-IN",
    "Hindi (हिन्दी)": "hi-IN",
    "Kannada (ಕನ್ನಡ)": "kn-IN",
    "Kashmiri (कॉशुर)": "ks-IN",
    "Konkani (कोंकणी)": "kok-IN",
    "Maithili (मैथिली)": "mai-IN",
    "Malayalam (മലയാളം)": "ml-IN",
    "Manipuri / Meitei (মৈতৈলোন্)": "mni-IN",
    "Marathi (मराठी)": "mr-IN",
    "Nepali (नेपाली)": "ne-IN",
    "Odia (ଓଡ଼ିଆ)": "or-IN",
    "Punjabi (ਪੰਜਾਬੀ)": "pa-IN",
    "Sanskrit (संस्कृतम्)": "sa-IN",
    "Santali (ᱥᱟᱱᱛᱟᱲᱤ)": "sat-IN",
    "Sindhi (سنڌي)": "sd-IN",
    "Tamil (தமிழ்)": "ta-IN",
    "Telugu (తెలుగు)": "te-IN",
    "Urdu (اردو)": "ur-IN"
}


language_name = st.selectbox(
    "Choose your language",
    list(language_options.keys())
)

language_code = language_options[language_name]

st.info("Record your farming question or upload an audio file.")

audio_file = st.file_uploader(
    "Upload your recorded question",
    type=["wav", "mp3", "m4a"]
)

if audio_file is not None:
    st.audio(audio_file)

    if st.button("🎤 Transcribe my question"):
        suffix = os.path.splitext(audio_file.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp_audio:
            temp_audio.write(audio_file.getvalue())
            audio_path = temp_audio.name

        try:
            with st.spinner("Understanding your question..."):
                transcript = speech_to_text(
                    audio_path,
                    language=language_code
                )

            st.session_state["transcript"] = transcript
            st.session_state["language_code"] = language_code

            st.success("Your question was transcribed!")
            st.write(transcript)

        except Exception as error:
            st.error(f"Transcription failed: {error}")

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

if "transcript" in st.session_state:
    st.subheader("Your question")
    st.write(st.session_state["transcript"])

    st.warning(
        "The agricultural answer engine will be connected here. "
        "No farming treatment has been generated yet."
    )
