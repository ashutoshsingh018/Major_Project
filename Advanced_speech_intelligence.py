"""
PRO VOICE ANALYST
=================
Version: 1.5 (High-Accuracy Translation Update)
Architecture: Streamlit + Whisper + HuggingFace Pipelines (NLLB-1.3B, T5, BART, Wav2Vec2)
Modules: ASR, Emotion, Sentiment, Toxicity, Summarization, Advanced Grammar, Zero-Shot Topic, S2S (gTTS)
"""

# ---------------------------------------------------------
# 0. SYSTEM IMPORTS
# ---------------------------------------------------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
# FIXED: Replaced sounddevice with audio_recorder for Cloud compatibility
from audio_recorder_streamlit import audio_recorder
import whisper
import os
import torch
import librosa
import nltk
import traceback
import scipy.signal
import re
from scipy.io.wavfile import write
from wordcloud import WordCloud
import pandas as pd
import io 
import datetime 
import difflib

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pro Voice Analyst (High-Res)",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UPDATED TITLE SECTION ---
st.markdown("""
<style>
:root {
    --title-color: #f0f0f0;
    --subtitle-color: #b5bac6;
    --divider-color: rgba(78, 168, 255, 0.45);
}

.nexus-container {
    text-align: center;
    margin-top: -32px;
}

.nexus-title {
    font-family: 'Helvetica Neue', sans-serif;
    font-weight: 850;
    font-size: 2.9rem;
    letter-spacing: -1.4px;
    margin: 0;
    color: var(--title-color);
    text-transform: uppercase;   /* FULL CAPS */
    text-shadow: 0 0 14px rgba(78,168,255,0.35);
}

.nexus-subtitle {
    font-family: 'Helvetica Neue', sans-serif;
    font-weight: 300;
    font-size: 1.05rem;
    letter-spacing: 4px;
    margin-top: 6px;
    color: var(--subtitle-color);
    opacity: 0.85;
}

.nexus-divider {
    width: 52%;
    margin: 18px auto 0 auto;
    border: none;
    border-top: 1.3px solid var(--divider-color);
    box-shadow: 0 0 6px rgba(78,168,255,0.25);
}
</style>

<div class="nexus-container">
    <h1 class="nexus-title">NEXUS OMEGA: ADVANCED SPEECH INTELLIGENCE</h1>
    <h3 class="nexus-subtitle">ACOUSTICS • EMOTION • ANALYTICS • INSIGHTS</h3>
    <hr class="nexus-divider">
</div>
""", unsafe_allow_html=True)


# HuggingFace & Transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS 

# Professional PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ---------------------------------------------------------
# NLTK SETUP
# ---------------------------------------------------------
for res in ['tokenizers/punkt_tab', 'tokenizers/punkt', 'corpora/stopwords']:
    try:
        if '/' in res: nltk.data.find(res)
    except LookupError:
        nltk.download(res.split('/')[-1], quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------
# UTILITY CLASS (For SystemLog)
# ---------------------------------------------------------
class SystemLog:
    @staticmethod
    def log(message, level="INFO"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{level}] SYSTEM: {message}")
        

# ---------------------------------------------------------
# AI MODELS LOADING
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Models (Switching to High-Accuracy NLLB 1.3B)…")
def load_models(model_size="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SystemLog.log(f"Loading Models on device: {device}")

    # Initialize Whisper for ASR 
    try:
        whisper_model = whisper.load_model(model_size, device=device)
    except Exception as e:
        SystemLog.log(f"Failed to load Whisper model: {e}", "ERROR")
        whisper_model = None


    # Load pipelines to the determined device
    def safe_pipeline_load(task, model_name, device):
        try:
            return pipeline(task, model=model_name, device=0 if device == "cuda" else -1)
        except Exception as e:
            SystemLog.log(f"Failed to load {model_name} for {task}: {e}", "ERROR")
            return None
            
    emotion_model = safe_pipeline_load("audio-classification", "superb/wav2vec2-base-superb-er", device)
    sentiment_model = safe_pipeline_load("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english", device)
    toxicity_model = safe_pipeline_load("text-classification", "unitary/unbiased-toxic-roberta", device)
    summary_model = safe_pipeline_load("summarization", "facebook/bart-large-cnn", device)
    
    # Initialize Zero-Shot Topic Classifier (BART)
    topic_model = safe_pipeline_load("zero-shot-classification", "facebook/bart-large-mnli", device)

    # NLLB Translator (UPGRADED to 1.3B for better Kannada Accuracy)
    try:
        # Changed from 600M to 1.3B to fix "useless" -> "useful" semantic errors
        nllb_model_name = "facebook/nllb-200-distilled-1.3B"
        tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
        translator = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name).to(device)
    except Exception as e:
        SystemLog.log(f"Failed to load NLLB translation models: {e}", "ERROR")
        tokenizer, translator = None, None

    # T5 Grammar Correction 
    try:
        grammar_tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
        grammar_model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction").to(device)
    except Exception as e:
        SystemLog.log(f"Failed to load T5 grammar models: {e}", "ERROR")
        grammar_tokenizer, grammar_model = None, None


    return (whisper_model, emotion_model, sentiment_model, toxicity_model, 
            summary_model, topic_model, translator, tokenizer, 
            grammar_model, grammar_tokenizer, device) 
            
# ---------------------------------------------------------
# AUDIO CLEANING & EMOTION FIX
# ---------------------------------------------------------
def clean_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        # Apply 10th order Butterworth High-pass filter at 100 Hz
        sos = scipy.signal.butter(10, 100, 'hp', fs=sr, output='sos')
        cleaned = scipy.signal.sosfilt(sos, y)
        cleaned = librosa.util.normalize(cleaned)
        write(audio_path, sr, cleaned)
    except:
        pass

def smart_emotion_detection(audio_path, model):
    if not model: return {"label": "Unavailable", "score": 0.0}

    clean_audio(audio_path)
    res = model(audio_path)
    res = sorted(res, key=lambda x: x["score"], reverse=True)
    top = res[0]

    # Heuristic: Downgrade low-confidence 'anger' to 'neutral'
    if top["label"] == "anger" and top["score"] < 0.60:
        neutral = next((x for x in res if x["label"] == "neutral"), None)
        if neutral:
            neutral["score"] = 0.55
            return neutral

    return top

# ---------------------------------------------------------
# NEW FEATURE FUNCTIONS (GRAMMAR & TTS)
# ---------------------------------------------------------

def advanced_grammar_check(text, model, tokenizer, device):
    """
    Advanced grammar correction using the T5 model.
    """
    if not all([text, model, tokenizer]):
        return text, 0, "", 100

    input_text = "grammar: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    try:
        # Beam search for better grammar optimization
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=8,
            early_stopping=True,
            temperature=0.7,
            repetition_penalty=2.0
        )

        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Compute difference ratio for scoring (measures dissimilarity)
        seq = difflib.SequenceMatcher(None, text, corrected_text)
        change_ratio = 1 - seq.ratio()
        
        # Levenshtein distance based score (40 is the minimum tolerance)
        grammar_score = max(40, 100 - (change_ratio * 100))

        # Highlight changed words
        diff = difflib.ndiff(text.split(), corrected_text.split())
        highlight = " ".join([
            f"**{token[2:]}**" if token.startswith("+ ") else token[2:]
            for token in diff if not token.startswith("- ")
        ])

        return corrected_text, change_ratio, highlight, round(grammar_score, 1)

    except Exception as e:
        SystemLog.log(f"T5 Grammar Check failed: {e.__class__.__name__}", "WARN")
        return text, 0, f"Error: {e.__class__.__name__}", 100 # Return original text on failure


def text_to_speech_file(text, lang="en"):
    """Converts text to audio for Speech-to-Speech, returns BytesIO stream."""
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        SystemLog.log(f"TTS Error ({lang}): {e}", "ERROR") 
        return None

def estimate_noise_level(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        # Estimate noise from the first 0.25s
        noise = np.mean(np.abs(y[:int(sr*0.25)]))
        speech = np.mean(np.abs(y))
        ratio = min(noise / (speech + 1e-5), 1.0) # Noise-to-Signal ratio

        if ratio < 0.2: level = "Low"
        elif ratio < 0.5: level = "Medium"
        else: level = "High"

        return level, ratio
    except:
        return "Unknown", 0

def calculate_wpm(text, audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        wpm = (len(text.split()) / duration) * 60
        return round(wpm, 1)
    except:
        return 0

def detect_speaking_time(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        # Split audio by silence (top_db=30 means anything 30dB below peak is silence)
        intervals = librosa.effects.split(y, top_db=30)
        seconds = sum((i[1] - i[0]) for i in intervals) / sr
        return round(seconds, 2)
    except:
        return 0

# --- FIX: ROBUST WORDCLOUD GENERATION ---
def generate_wordcloud(text):
    if not text or len(text.strip()) == 0:
        return None
    try:
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        return wc.to_array()
    except ValueError:
        # Handles empty vocab error
        return None
    except Exception:
        return None

def classify_topic(text, model):
    """Classifies the topic using a Zero-Shot Classification model."""
    if not model or len(text.split()) < 5:
        return {"label": "Too Short/Unavailable", "score": 0.0, "raw_results": []}

    candidate_labels = [
        "Business and Finance", 
        "Technology and Computing", 
        "Health and Wellness", 
        "Politics and Society",
        "Sports and Entertainment", 
        "Education and Learning",
        "Personal/General Conversation"
    ]
    
    try:
        result = model(text, candidate_labels)
        
        top_topic = result['labels'][0]
        top_score = result['scores'][0]
        
        raw_results = list(zip(result['labels'], result['scores']))
        
        return {
            "label": top_topic,
            "score": round(top_score * 100, 1),
            "raw_results": raw_results
        }
    except Exception as e:
        SystemLog.log(f"Topic Classification failed: {e.__class__.__name__}", "WARN")
        return {"label": f"Error: {e.__class__.__name__}", "score": 0.0, "raw_results": []}


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "target_lang_select" not in st.session_state:
    st.session_state.target_lang_select = "English"

# Callback for selectbox (does nothing, just prevents immediate rerun)
def update_target_lang():
    pass
# ---------------------------------------------------------
# FILE SAVE / RECORDING FUNCTIONS
# ---------------------------------------------------------
# REMOVED: record_audio_filesafe (sounddevice dependency)

def save_uploaded_file(uploaded_file):
    try:
        tmp = tempfile.gettempdir()
        path = os.path.join(tmp, f"upload_{os.getpid()}_{uploaded_file.name}")
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return path
    except:
        return None


# ---------------------------------------------------------
# SIDEBAR UI (UPDATED WITH KANNADA & NLLB)
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Whisper Model Size:", 
        ["base (Fast)", "small (Balanced)", "medium (High Accuracy)"],
        index=2
    )
    model_size = model_choice.split()[0]

    # UNPACKING MODELS
    (
        whisper_model,
        emotion_model,
        sentiment_model,
        toxicity_model,
        summary_model,
        topic_model,
        translator,
        tokenizer,
        grammar_model,
        grammar_tokenizer,
        device_type
    ) = load_models(model_size)

    # Check for model availability
    if not whisper_model: st.error("Whisper Model Failed to Load.")
    if not all([translator, tokenizer]): st.error("NLLB Translator Failed to Load.")
    if not all([grammar_model, grammar_tokenizer]): st.error("T5 Grammar Failed to Load.")


    st.divider()

    st.subheader("🗣️ Language Settings")
    translate_direct = st.checkbox("Transcribe directly to English", value=False)

    # --- UPDATED LANGUAGE LIST (INCLUDING KANNADA) ---
    lang_choice = st.selectbox(
        "Spoken Language:",
        ["Auto-Detect", "English", "Hindi", "Kannada", "Tamil", "Telugu", "Marathi", "Bengali", "Spanish", "Chinese"]
    )

    whisper_langs = {
        "Auto-Detect": None,
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn", # KANNADA WHISPER CODE
        "Tamil": "ta",
        "Telugu": "te",
        "Marathi": "mr",
        "Bengali": "bn",
        "Spanish": "es",
        "Chinese": "zh",
    }
    selected_lang = whisper_langs[lang_choice]

    # Adding initial prompts for better Indian language recognition
    prompts = {
        "hi": "नमस्ते, मैं हिंदी बोल रहा हूँ।",
        "en": "Hello, this is a standard English sentence.",
        "es": "Hola, esta es una frase en español.",
        "zh": "你好，这是中文句子。",
        "kn": "ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ವಾಕ್ಯ.", # KANNADA PROMPT
        "ta": "வணக்கம், இது தமிழ் வாக்கியம்.", 
        "te": "నమస్కారం, ఇది తెలుగు వాక్యం.", 
        "mr": "नमस्कार, हे मराठी वाक्य आहे.", 
        "bn": "নমস্কার, এটি বাংলা বাক্য।", 
    }
    initial_prompt = prompts.get(selected_lang, None)

    st.divider()

    st.subheader("🎧 Input Method")
    input_method = st.radio("Choose Input:", ["🎤 Record", "📁 Upload"])

    if input_method == "🎤 Record":
        st.markdown("Click the microphone to record:")
        # This component handles the recording in the browser
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
        )

        if audio_bytes:
            # Save the recorded bytes to a temporary file
            try:
                tmp = tempfile.gettempdir()
                path = os.path.join(tmp, f"record_{os.getpid()}.wav")
                with open(path, "wb") as f:
                    f.write(audio_bytes)
                
                st.session_state.audio_path = path
                st.session_state.analysis_results = None
                st.success("Audio recorded successfully! Click 'Analyze Audio' to proceed.")
                
            except Exception as e:
                st.error(f"Error saving recording: {e}")

    else:
        uploaded = st.file_uploader("Upload WAV/MP3/FLAC File:", type=["wav", "mp3", "flac"]) 
        if uploaded:
            path = save_uploaded_file(uploaded)
            if path:
                st.session_state.audio_path = path


# ---------------------------------------------------------
# MAIN PROCESSING LOGIC
# ---------------------------------------------------------
if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):

    st.audio(st.session_state.audio_path)

    if st.button("🚀 Analyze Audio"):
        # Pre-check for essential models
        if not all([whisper_model, emotion_model, sentiment_model, toxicity_model, summary_model, topic_model, translator, tokenizer, grammar_model, grammar_tokenizer]):
             st.error("Some required AI models failed to load. Please restart the app or check console logs.")
             st.stop()

        with st.spinner("AI is analyzing your speech…"):
            try:
                task = "translate" if translate_direct else "transcribe"

                # 1. WHISPER CALL
                result = whisper_model.transcribe(
                    st.session_state.audio_path,
                    fp16=False,
                    language=selected_lang,
                    task=task,
                    initial_prompt=initial_prompt,
                )

                text = result["text"].strip()
                # Clean up any residual bracketed text (whisper artifacts)
                text = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>', '', text).strip()


                if not text:
                    st.warning("No speech detected.")
                    st.stop()
                else:
                    # 2. EMOTION
                    em = smart_emotion_detection(st.session_state.audio_path, emotion_model)

                    # 3. SENTIMENT (only English)
                    try:
                        sent = sentiment_model(text)[0]
                    except:
                        sent = {"label": "NEUTRAL", "score": 0.5}

                    # 4. TOXICITY (only English)
                    try:
                        tox_raw = toxicity_model(text)
                        tox = tox_raw[0][0] if isinstance(tox_raw, list) else tox_raw[0]
                    except:
                        tox = {"label": "safe", "score": 0.0}

                    # 5. SUMMARY
                    wc = len(text.split())
                    if wc < 20:
                        summary = "Original text is too short to summarize."
                    else:
                        try:
                            summary = summary_model(
                                text,
                                max_length=int(wc * 0.8),
                                min_length=5,
                                do_sample=False
                            )[0]["summary_text"]
                        except:
                            summary = "Summary unavailable."

                    # 6. KEYWORDS
                    toks = word_tokenize(text.lower())
                    stops = set(stopwords.words("english"))
                    kws = [w for w in toks if w.isalnum() and w not in stops]

                    from collections import Counter
                    top_kws = [k for k, v in Counter(kws).most_common(5)]

                    # 7. ACOUSTIC & SPEED METRICS
                    noise_label, noise_ratio = estimate_noise_level(st.session_state.audio_path)
                    wpm = calculate_wpm(text, st.session_state.audio_path)
                    speaking_time = detect_speaking_time(st.session_state.audio_path)
                    
                    # 8. GRAMMAR (T5 CHECK)
                    corrected_text, change_ratio, highlight_text, grammar_score = advanced_grammar_check(
                        text, grammar_model, grammar_tokenizer, device_type
                    )

                    grammar_results = {
                        "score": grammar_score,
                        "changes": change_ratio,
                        "status": "AI Corrected" if change_ratio > 0.03 else "Excellent",
                        "corrected_text": corrected_text,
                        "highlight": highlight_text
                    }
                    
                    # 9. TOPIC (ZERO-SHOT)
                    topic_results = classify_topic(text, topic_model)


                    # 10. Communication Score
                    # Weighted formula for overall quality
                    comm_score = (
                        (1 - noise_ratio) * 30 +
                        max(0, 1 - abs(wpm - 140) / 140) * 30 +
                        (1 - tox["score"]) * 20 +
                        (0.5 if em["label"] in ["neutral", "happy"] else 0.2) * 20
                    )
                    comm_score = round(comm_score, 1)

                    # SAVE RESULTS
                    st.session_state.analysis_results = {
                        "text": text,
                        "emotion": em,
                        "sentiment": sent,
                        "toxicity": tox,
                        "summary": summary,
                        "keywords": top_kws,
                        "lang": lang_choice,

                        # acoustic/speed metrics
                        "noise": (noise_label, noise_ratio),
                        "wpm": wpm,
                        "speaking_time": speaking_time,
                        "score": comm_score,
                        
                        # text metrics
                        "grammar": grammar_results,
                        "topic": topic_results,
                    }
                    st.toast("Analysis Complete!", icon="✅")

            except Exception as e:
                st.error("Analysis Failed! One of the pipelines encountered an error.")
                st.code(traceback.format_exc())
# ---------------------------------------------------------
# DASHBOARD VISUALIZATION
# ---------------------------------------------------------
if st.session_state.analysis_results:

    data = st.session_state.analysis_results
    st.divider()
    
    # ---------------------------------------------------------
    # ROW 1: CORE NLP METRICS 
    # ---------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    # EMOTION CARD
    emo = data["emotion"]["label"].upper()
    emo_score = data["emotion"]["score"]
    color = "color-neutral"
    if emo in ["HAPPY", "NEUTRAL"]:
        color = "color-safe"
    elif emo in ["ANGER", "SADNESS"]:
        color = "color-danger"

    c1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Emotion</div>
        <div class="metric-value {color}">{emo}</div>
        <div>{emo_score*100:.1f}% Confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # SENTIMENT CARD
    sent_label = data["sentiment"]["label"]
    sent_col = "color-safe" if sent_label == "POSITIVE" else "color-danger"

    c2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Sentiment (English Only)</div>
        <div class="metric-value {sent_col}">{sent_label}</div>
        <div>{data['sentiment']['score']:.2f} Score</div>
    </div>
    """, unsafe_allow_html=True)

    # TOXICITY CARD
    tox_score = data["toxicity"]["score"]
    tox_label = "Toxic" if tox_score > 0.5 else "Safe"
    tox_color = "color-danger" if tox_score > 0.5 else "color-safe"

    c3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Toxicity (English Only)</div>
        <div class="metric-value {tox_color}">{tox_label}</div>
        <div>{tox_score:.2f} Prob</div>
    </div>
    """, unsafe_allow_html=True)

    # WORD COUNT CARD
    wc = len(data["text"].split())
    c4.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Word Count</div>
        <div class="metric-value color-neutral">{wc}</div>
        <div>Language: {data["lang"]}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # ROW 2: ACOUSTIC & QUALITY METRICS 
    # ---------------------------------------------------------
    
    c5, c6, c7 = st.columns(3)
    
    # Grouped card 1: Noise & WPM
    c5.markdown(f"""
    <div class="metric-card" style='height: 120px;'>
        <div class="metric-label">Noise / WPM</div>
        <div class="metric-value">{data['noise'][0]} / {data['wpm']} WPM</div>
        <div>Ratio: {data['noise'][1]:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Grouped card 2: Topic & Grammar
    topic_label = data['topic']['label']
    topic_score = data['topic']['score']
    grammar_score = data['grammar']['score']
    
    c6.markdown(f"""
    <div class="metric-card" style='height: 120px;'>
        <div class="metric-label">Topic / Grammar</div>
        <div class="metric-value">{topic_label}</div>
        <div>Confidence: {topic_score:.1f}% | Grammar: {grammar_score:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Grouped card 3: Speaking Time & Final Score
    c7.markdown(f"""
    <div class="metric-card" style='height: 120px;'>
        <div class="metric-label">Active Time / Score</div>
        <div class="metric-value">{data['speaking_time']} sec</div>
        <div>Comm Score: {data['score']}/100</div>
    </div>
    """, unsafe_allow_html=True)


    st.divider()

    # ---------------------------------------------------------
    # MAIN TABS
    # ---------------------------------------------------------
    t1, t2, t3, t4, t5 = st.tabs([
        "📝 Transcript",
        "🌍 Translate",
        "📉 Emotion Timeline",
        "📄 Pro PDF",
        "☁️ Word Cloud"
    ])

    # ---------------------------------------------------------
    # TAB 1 — TRANSCRIPT
    # ---------------------------------------------------------
    with t1:
        st.subheader("Original Text")
        st.info(data["text"])

        st.subheader("AI Summary")
        st.write(data["summary"])
        
        st.divider()

        # --- TOPIC ANALYSIS SECTION ---
        st.subheader("🧠 Topic Analysis")
        topic_data = data['topic']
        
        st.metric(
            label="Most Likely Topic",
            value=topic_data['label'],
            delta=f"{topic_data['score']}% Confidence"
        )
        
        if topic_data['raw_results']:
            with st.expander("View All Topic Scores"):
                topic_df = pd.DataFrame(topic_data['raw_results'], columns=['Topic', 'Score'])
                topic_df['Score'] = (topic_df['Score'] * 100).round(1).astype(str) + '%'
                st.dataframe(topic_df, hide_index=True, use_container_width=True)

        st.divider()
        # --- END TOPIC ANALYSIS SECTION ---

        # --- GRAMMAR SECTION ---
        g_data = data['grammar']
        st.subheader("7️⃣ Advanced Grammar Correction (T5 - English Only)")
        
        st.markdown(f"**T5 Corrected Text:**")
        st.code(g_data['corrected_text'], language='text')
        st.subheader("Grammar Changes Highlighted")
        st.markdown(g_data["highlight"])


        g_col1, g_col2 = st.columns(2)
        
        g_col1.metric(
            label="Grammar Quality Score (AI Index)",
            value=f"{g_data['score']}%",
            delta=f"{g_data['changes']} major change(s)",
        )
        g_col2.markdown(f"**Status:** <span style='color: {'#2ecc71' if g_data['score'] >= 90 else ('#f39c12' if g_data['score'] >= 75 else '#e74c3c')}'>{g_data['status']}</span>", unsafe_allow_html=True)

        st.divider()
        # --- END GRAMMAR SECTION ---


        st.subheader("Top Keywords")
        st.success(", ".join(data["keywords"]))


    # ---------------------------------------------------------
    # TAB 2 — TRANSLATION (UPDATED WITH KANNADA & NLLB)
    # ---------------------------------------------------------
    with t2:
        if not all([translator, tokenizer]):
            st.error("NLLB Translator is unavailable. Check model loading logs.")
        else:
            st.subheader("Text & Speech-to-Speech Translation (NLLB-200)")
            
            # --- NLLB FLORES-200 LANGUAGE CODES ---
            # These are specific codes used by NLLB. 
            # Note: "kan_Knda" is Kannada.
            nllb_lang_map = {
                "English": "eng_Latn",
                "Hindi": "hin_Deva",
                "Kannada": "kan_Knda", # KANNADA NLLB CODE
                "Tamil": "tam_Taml",
                "Telugu": "tel_Telu",
                "Marathi": "mar_Deva",
                "Bengali": "ben_Beng",
                "Spanish": "spa_Latn",
                "German": "deu_Latn",
                "French": "fra_Latn",
                "Chinese": "zho_Hans"
            }
            
            # --- GTTS LANGUAGE CODES (Updated with Kannada) ---
            gtts_lang_map = {
                "English": "en", "Hindi": "hi", "Kannada": "kn", "Tamil": "ta", "Telugu": "te", 
                "Marathi": "mr", "Bengali": "bn", "Spanish": "es", 
                "German": "de", "French": "fr", "Chinese": "zh-cn"
            }

            st.selectbox(
                "Target Language:", 
                list(nllb_lang_map.keys()),
                key="target_lang_select",
                on_change=update_target_lang
            )
            
            selected_target_lang = st.session_state.target_lang_select
            
            if st.button("Translate & Generate Audio", key="translate_button"):
                with st.spinner("Translating via NLLB and generating speech…"):
                    
                    try:
                        # 1. NLLB Text Translation
                        # Tokenize input
                        inputs = tokenizer(data["text"], return_tensors="pt").to(device_type)
                        
                        # Get Target Code
                        target_code = nllb_lang_map[selected_target_lang]
                        
                        # Generate translation 
                        # FIX: ADDED BEAM SEARCH & REPETITION PENALTY FOR ACCURACY
                        translated_tokens = translator.generate(
                            **inputs,
                            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
                            max_length=512,
                            num_beams=5,             # Increases accuracy
                            repetition_penalty=1.2    # Reduces hallucinations/loops
                        )
                        
                        final_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                        st.success(final_text)

                        # 2. SPEECH-TO-SPEECH (gTTS)
                        st.subheader("🔊 Listen to Translation")
                        
                        tts_lang_code = gtts_lang_map.get(selected_target_lang, "en") 
                        
                        audio_stream = text_to_speech_file(final_text, lang=tts_lang_code)
                        
                        if audio_stream:
                            st.audio(audio_stream, format="audio/mp3")
                        else:
                            st.warning(f"Could not generate audio in {selected_target_lang} (TTS Failed).")
                            SystemLog.log(f"TTS Failed for {selected_target_lang}", "WARN")

                    except Exception as e:
                        st.error(f"Translation Error: {e.__class__.__name__}")
                        SystemLog.log(f"Translation Error: {traceback.format_exc()}", "ERROR")

    # ---------------------------------------------------------
    # TAB 3 — EMOTION TIMELINE
    # ---------------------------------------------------------
    with t3:
        st.subheader("Emotion Flow Over Time")
        if emotion_model:
            st.warning("Note: Timeline analysis is resource-intensive and is capped at 60 seconds.")
            if st.button("Generate Timeline"):
                with st.spinner("Processing audio chunks…"):
                    # 
                    
                    y, sr = librosa.load(st.session_state.audio_path, sr=16000)
                    step = int(1.0 * sr)  # 1 sec
                    timeline = []

                    for i in range(0, min(len(y), 60 * sr), step):
                        chunk = y[i:i + step]
                        if len(chunk) > sr * 0.5:
                            tf = tempfile.mktemp(suffix=".wav")
                            write(tf, sr, chunk)
                            try:
                                # Get the top confidence score from the chunk
                                score = emotion_model(tf)[0]["score"]
                                timeline.append(score)
                            except:
                                timeline.append(0.0)
                            finally:
                                if os.path.exists(tf):
                                    os.remove(tf)

                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(timeline, linewidth=2, color="#3498db")
                    ax.set_title("Emotional Intensity (Per Second)")
                    ax.set_ylabel("Top Emotion Score")
                    ax.set_xlabel("Time (s)")
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    st.pyplot(fig)
        else:
            st.error("Emotion model is unavailable. Timeline cannot be generated.")


    # ---------------------------------------------------------
    # TAB 4 — PROFESSIONAL PDF REPORT
    # ---------------------------------------------------------
    with t4:
        st.subheader("Generate Professional Report")

        def generate_pro_pdf(data):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_file:
                doc = SimpleDocTemplate(pdf_file.name, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []

                title_style = ParagraphStyle(
                    'Title',
                    parent=styles['Heading1'],
                    alignment=1,
                    fontSize=24,
                    textColor=colors.darkblue
                )
                story.append(Paragraph("Speech Analysis Report", title_style))
                story.append(Spacer(1, 20))

                # Updated Table Data
                table_data = [
                    ["Metric", "Result", "Score"],
                    ["Topic", data["topic"]["label"], f"{data['topic']['score']:.1f}%"],
                    ["Emotion", data["emotion"]["label"].upper(), f"{data['emotion']['score']:.2f}"],
                    ["Sentiment", data["sentiment"]["label"], f"{data['sentiment']['score']:.2f}"],
                    ["Toxicity", "Toxic" if data["toxicity"]["score"] > 0.5 else "Safe", f"{data['toxicity']['score']:.2f}"],
                    ["Noise", data["noise"][0], f"{data['noise'][1]:.2f}"],
                    ["WPM", data["wpm"], "Speed"],
                    ["Speaking Time", data["speaking_time"], "Seconds"],
                    ["Grammar Score (T5)", data["grammar"]["score"], "AI Index"],
                    ["Communication Score", data["score"], "AI Index"],
                ]

                table = Table(table_data, colWidths=[150, 150, 150])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 20))

                story.append(Paragraph("<b>Transcript (Language: " + data["lang"] + "):</b>", styles['Heading3']))
                story.append(Paragraph(data["text"], styles["BodyText"]))
                
                story.append(Paragraph("<b>T5 Grammar Correction:</b>", styles['Heading3']))
                story.append(Paragraph(data["grammar"]["corrected_text"], styles["BodyText"]))
                
                story.append(Spacer(1, 12))

                story.append(Paragraph("<b>Summary:</b>", styles['Heading3']))
                story.append(Paragraph(data["summary"], styles["BodyText"]))

                doc.build(story)
                return pdf_file.name

        if st.button("Generate PDF"):
            pdf_path = generate_pro_pdf(data)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, "Speech_Report.pdf")


    # ---------------------------------------------------------
    # TAB 5 — WORD CLOUD
    # ---------------------------------------------------------
    with t5:
        st.subheader("Word Cloud")
        # 
        img = generate_wordcloud(data["text"])
        if img is not None:
            # FIX: Updated deprecated use_column_width=True to "auto"
            st.image(img, use_container_width=True)
        else:
            st.warning("Not enough distinct words found to generate a Word Cloud.")

# ---------------------------------------------------------
# GLOBAL STYLING (CSS - DARK MODE COMPATIBLE)
# ---------------------------------------------------------
st.markdown("""
<style>

html, body {
    margin: 0;
    padding: 0;
}

/* Updated Metric Card to use Streamlit Native CSS Variables for Dark Mode */
.metric-card {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    margin-bottom: 14px;
    height: 100%; /* Ensure uniform height */
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.1);
}

.metric-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-color);
    opacity: 0.7;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 24px; /* Slightly adjusted for better mobile view */
    font-weight: 700;
    margin: 6px 0;
    color: var(--text-color);
}

.color-safe { color: #2ecc71 !important; }
.color-danger { color: #e74c3c !important; }
.color-neutral { color: #3498db !important; }
.color-warn { color: #f39c12 !important; }

.stButton>button {
    width: 100%;
    border-radius: 8px !important;
    font-weight: bold !important;
    height: 50px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# OPTIONAL: AUTO-CORRECT TOGGLE
# ---------------------------------------------------------
st.sidebar.subheader("📝 Extra Settings")

apply_autocorrect = st.sidebar.checkbox("Apply T5 auto-correct to transcript (Overrides analysis text)", value=False)

if apply_autocorrect and st.session_state.analysis_results:
    # Ensure T5 models are available before attempting to use them
    if not all([grammar_model, grammar_tokenizer]):
        st.sidebar.error("T5 models are unavailable for auto-correct.")
    else:
        # Re-run the correction logic if the original text differs (which it might due to transcription variability)
        original_text = st.session_state.analysis_results.get("text_before_autocorrect", st.session_state.analysis_results["text"])
        
        corrected_text, change_ratio, highlight_text, grammar_score = advanced_grammar_check(
            original_text, grammar_model, grammar_tokenizer, device_type
        )
        
        # Save the original text just in case the user toggles back
        if "text_before_autocorrect" not in st.session_state.analysis_results:
            st.session_state.analysis_results["text_before_autocorrect"] = st.session_state.analysis_results["text"]
            
        st.session_state.analysis_results["text"] = corrected_text
        st.sidebar.success("Auto-correct applied ✔")
elif not apply_autocorrect and st.session_state.analysis_results and "text_before_autocorrect" in st.session_state.analysis_results:
    # Revert to original text if the toggle is switched off
    st.session_state.analysis_results["text"] = st.session_state.analysis_results.pop("text_before_autocorrect")
    st.sidebar.info("Reverted to original transcript.")


# The APP IS COMPLETE ✔
st.success("✔ Application Loaded Successfully (v1.5 with High-Res Translation & Dark Mode Support)")