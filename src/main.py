import streamlit as st
import numpy as np
from PIL import Image
import os
import tempfile

from preprocessing.processor import ImagePreprocessor
from ocr.engine import HeritageOCREngine
from agent.agent import HeritageAgent
from ledger.ledger import BlockchainLedger

# Page Config
st.set_page_config(layout="wide", page_title="Heritage OCR & AI Agent")

# CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .reportview-container {
        background: #f0f2f6
    }
</style>
""", unsafe_allow_html=True)

st.title("📜 Heritage OCR: AI for Indian Archives")
st.markdown("Digitize, Restore, and Converse with 100-year-old Land Records and Manuscripts.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    st.header("Configuration")
    
    # Check if API Key is in Secrets (Cloud Deployment) or Sidebar (Local/Manual)
    if "OPENAI_API_KEY" in st.secrets:
        st.success("API Key loaded from Cloud Secrets ✅")
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("OpenAI API Key", type="password", help="Needed for the AI Agent")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
    ocr_mode = st.selectbox("OCR Engine", ["Doctr (Deep Learning)", "Tesseract (Fallback)"])
    indic_support = st.checkbox("Enable Indic Script Fallback", value=True)
    use_preprocessing = st.checkbox("Enable Advanced Preprocessing (Deskew/Denoise)", value=False, help="Enable if document is very noisy or rotated. Disable for clean scans.")
    
    st.divider()
    st.markdown("### About")
    st.info("Built for Vibecraft Hackathon using Doctr & OpenAI.\n\nTeam: Vishanth Dandu, Justin Joshua, Bhargav Gutta, Nikhil Rajendra")

# --- Initialization ---

if 'ocr_engine' not in st.session_state:
    try:
        # Load CPU for demo to avoid CUDA issues in random envs, or check availability
        st.session_state.ocr_engine = HeritageOCREngine(use_gpu=False) 
        st.session_state.preprocessor = ImagePreprocessor()
        st.session_state.agent = HeritageAgent()
        st.session_state.ledger = BlockchainLedger()
    except Exception as e:
        st.error(f"Failed to load models: {e}")

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Main Interface ---

uploaded_file = st.file_uploader("Upload a Document Image", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file:
    # Load and process
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Document")
        st.image(image, use_column_width=True)
        
        process_btn = st.button("🚀 Process Document", use_container_width=True)

    with col2:
        st.subheader("Digitized Output")
        
        if process_btn:
            # Default to original
            ocr_input = image_np
            
            if use_preprocessing:
                with st.spinner("Preprocessing (Denoising, Deskewing)..."):
                    preprocessed = st.session_state.preprocessor.process_pipeline(image_np)
                    st.image(preprocessed['enhanced'], caption="Enhanced (CLAHE)", channels="GRAY", width=300)
                    # Use deskewed for OCR
                    ocr_input = preprocessed['deskewed']
            else:
                st.info("ℹ️ Preprocessing skipped (using raw image). Enable in sidebar if needed.")
                preprocessed = {"enhanced": image_np, "deskewed": image_np} # Fallback for dict

            with st.spinner("Running OCR..."):
                # Doctr works best on RGB
                ocr_result = st.session_state.ocr_engine.detect_and_recognize(
                    ocr_input, 
                    use_tesseract_fallback=indic_support
                )
                
            with st.spinner("AI Agent Structuring..."):
                 structured_data = st.session_state.agent.process_document(ocr_result['full_text'])
            
            # --- Blockchain Registration ---
            with st.spinner("Hashing & Registering on Ledger..."):
                 ledger_entry = st.session_state.ledger.add_record(structured_data, metadata={"source": uploaded_file.name})

            st.session_state.processed_data = {
                "text": ocr_result['full_text'],
                "structured": structured_data,
                "images": preprocessed,
                "ledger": ledger_entry
            }
            
        if st.session_state.processed_data:
            # --- Blockchain Badge ---
            ledger_info = st.session_state.processed_data.get('ledger')
            if ledger_info:
                st.success(f"✅ **Blockchain Verified** | Hash: `{ledger_info['hash'][:15]}...` | Index: #{ledger_info['index']}")
                with st.expander("⛓️ View Audit Trail"):
                    st.json(ledger_info)
                    st.info("ℹ️ This hash proves the document existed in this state at this time.")

            tabs = st.tabs(["Extracted Text", "Structured Data", "Translation"])
            
            with tabs[0]:
                st.text_area("Raw Text", st.session_state.processed_data['text'], height=300)
                
            with tabs[1]:
                st.json(st.session_state.processed_data['structured'])
                
            with tabs[2]:
                target_lang = st.selectbox("Translate to", ["Hindi", "Telugu", "Tamil", "English"])
                if st.button("Translate"):
                    with st.spinner("Translating..."):
                         trans = st.session_state.agent.chat(
                             st.session_state.processed_data['text'], 
                             f"Translate the full text into {target_lang}. Preserve formatting.",
                             target_language=target_lang
                         )
                         st.markdown(trans)

# --- Chat Interface ---
st.divider()
st.subheader("🤖 Chat with your Document (Vernacular Support)")

if st.session_state.processed_data:
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something about the document (e.g., 'Who is the owner?')"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.chat(
                    st.session_state.processed_data['text'], 
                    prompt
                )
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
else:
    st.info("Please upload and process a document to start chatting.")
