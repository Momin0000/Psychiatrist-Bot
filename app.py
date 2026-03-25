import streamlit as st
from transformers import pipeline
import time
import json
import os

# ----------------------
# Page setup
# ----------------------
st.set_page_config(page_title="PsyChat AI Brain", page_icon="🧠", layout="centered")
st.title("🧠 PsyChat — Layered AI Brain Prototype")
st.caption("Emotion → Reflection → Question → Strategy, adaptive flow")

# ----------------------
# Load emotion detection model
# ----------------------
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

emotion_model = load_emotion_model()

# ----------------------
# Load strategies from JSON
# ----------------------
def load_strategies():
    path = "strategies.json"
    if not os.path.exists(path):
        st.error("⚠️ strategies.json not found in the project folder.")
        return {}
    with open(path, "r") as f:
        return json.load(f)

STRATEGIES = load_strategies()

# ----------------------
# Session state
# ----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = None
if "stage" not in st.session_state:
    st.session_state.stage = "reflection"  # reflection → question → strategy
if "memory" not in st.session_state:
    st.session_state.memory = []

# ✅ Initialize tracking for used strategies & reflections
if "strategies" not in st.session_state:
    st.session_state.strategies = {}
if "reflections" not in st.session_state:
    st.session_state.reflections = {}

# ----------------------
# Helper: rotate through items
# ----------------------
def get_next_item(emotion, item_type):
    pool = STRATEGIES.get(emotion, {}).get(item_type, [])
    if not pool:
        return None
    used = st.session_state[item_type].get(emotion, [])
    available = [x for x in pool if x not in used]
    if not available:
        st.session_state[item_type][emotion] = []  # reset after all used
        available = pool
    choice = available[0]  # pick first unused
    st.session_state[item_type].setdefault(emotion, []).append(choice)
    return choice

# ----------------------
# Bot logic
# ----------------------
def bot_logic(user_input):
    # Detect dominant emotion
    results = emotion_model(user_input)[0]
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_emotion = results[0]['label'].lower()

    # Save to memory
    st.session_state.memory.append({"text": user_input, "emotion": top_emotion})

    # Conversation stage
    stage = st.session_state.stage
    reply = ""

    if stage == "reflection":
        reply = f"I hear you. It sounds like you may be feeling **{top_emotion}**."
        st.session_state.stage = "question"

    elif stage == "question":
        question = get_next_item(top_emotion, "questions") or "Can you tell me a bit more about this feeling?"
        reply = question
        st.session_state.stage = "strategy"

    elif stage == "strategy":
        tip = get_next_item(top_emotion, "strategies") or "Try a mindful pause and notice your breath."
        reply = f"Here’s something that might help with {top_emotion}: {tip}"
        st.session_state.stage = "reflection"

    st.session_state.last_emotion = top_emotion
    return reply

# ----------------------
# Render chat
# ----------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Type your message…")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            time.sleep(0.5)
            response = bot_logic(user_text)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.caption("Prototype: Layered AI brain with adaptive reflection → question → strategy. Research-only, not medical advice.")
