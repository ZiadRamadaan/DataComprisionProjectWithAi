import streamlit as st
import os
import bz2
import pickle
import re
from collections import Counter

# Shannon-Fano Algorithm
def shannon_fano_sort(symbols):
    return sorted(symbols, key=lambda x: x[1], reverse=True)

def split_symbols(symbols):
    total = sum(freq for _, freq in symbols)
    acc = 0
    for i in range(len(symbols)):
        acc += symbols[i][1]
        if acc >= total / 2:
            return symbols[:i+1], symbols[i+1:]
    return symbols, []

def assign_codes(symbols, code='', codebook=None):
    if codebook is None:
        codebook = {}
    if len(symbols) == 1:
        codebook[symbols[0][0]] = code
        return codebook
    left, right = split_symbols(symbols)
    assign_codes(left, code + '0', codebook)
    assign_codes(right, code + '1', codebook)
    return codebook

def build_shannon_fano_codes(freq_dict):
    sorted_symbols = shannon_fano_sort(list(freq_dict.items()))
    return assign_codes(sorted_symbols)

def get_most_frequent_words(text, top_n=10):
    words = re.findall(r'\w+', text.lower())
    freq = Counter(words)
    _ = build_shannon_fano_codes(freq)
    return [word for word, _ in freq.most_common(top_n)]

# Shannon-Fano Search Model
class ShannonFanoSearchModel:
    def __init__(self):
        self.data = []
        self.keywords = []

    def train(self, text):
        articles = re.split(r'[\n\r]{2,}', text)
        self.data = [article.strip() for article in articles if article.strip()]
        all_text = ' '.join(self.data)
        self.keywords = get_most_frequent_words(all_text)

    def generate(self, prompt, max_points=5):
        prompt_words = set(re.findall(r'\w+', prompt.lower()))
        matching_sentences = []

        for article in self.data:
            sentences = re.split(r'[.؟!\n\r]+', article)
            for sentence in sentences:
                sentence_words = set(re.findall(r'\w+', sentence.lower()))
                if prompt_words & sentence_words:
                    cleaned = sentence.strip()
                    if len(cleaned) > 10:
                        matching_sentences.append(cleaned)

        unique_sentences = list(dict.fromkeys(matching_sentences))
        return unique_sentences[:max_points] if unique_sentences else ["No relevant points found."]

    def save(self, filename="chat_data.pbz2"):
        data = pickle.dumps((self.data, self.keywords))
        with bz2.BZ2File(filename, 'wb') as f:
            f.write(data)

    def load(self, filename="chat_data.pbz2"):
        with bz2.BZ2File(filename, 'rb') as f:
            self.data, self.keywords = pickle.loads(f.read())

# Streamlit UI
st.set_page_config(page_title="Smart Assistant (Shannon-Fano)", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
        .stButton>button {background-color: #4CAF50; color: white; font-weight: bold; border-radius: 8px;}
        .stTextInput>div>div>input {border-radius: 8px;}
        .stTextArea>div>textarea {border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

st.title("Smart Text Assistant using Shannon-Fano")

model_path = "chat_data.pbz2"

if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

if "model" not in st.session_state:
    st.session_state.model = ShannonFanoSearchModel()

# Upload and train section
st.subheader("Upload and Train on a Text File")
file = st.file_uploader("Upload a .txt file", type="txt")

if file:
    text = file.read().decode("utf-8")
    st.text_area("Text Preview:", text[:1000], height=150)
    if st.button("Train Model"):
        st.session_state.model.train(text)
        st.session_state.model.save(model_path)
        st.session_state.model_ready = True
        st.success("Model trained and data saved successfully.")
elif os.path.exists(model_path) and not st.session_state.model_ready:
    try:
        st.session_state.model.load(model_path)
        st.session_state.model_ready = True
    except Exception as e:
        st.error(f"Error loading existing data: {e}")

# Chat interface
st.subheader("Ask Your Assistant")
if st.session_state.model_ready:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    with st.form(key='chat_form', clear_on_submit=True):
        prompt = st.text_input("📝 Enter a keyword or topic:")
        submit_button = st.form_submit_button("Send")

    if submit_button and prompt:
        responses = st.session_state.model.generate(prompt)
        st.session_state.chat.append(("You", prompt))
        for r in responses:
            st.session_state.chat.append(("Assistant", "• " + r))

    for sender, msg in reversed(st.session_state.chat):
        st.markdown(
            f"<div style='padding:10px;border-radius:10px;margin-bottom:5px'>"
            f"<b>{sender}:</b> {msg}</div>",
            unsafe_allow_html=True
        )
else:
    st.info("Please upload and train the model using a text file first.")
