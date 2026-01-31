# Heritage OCR - Vibecraft Hackathon

A specialized AI system for digitizing, restoring, and querying heritage documents (Land records, manuscripts) with a focus on Indic scripts and vernacular languages.

## 🌟 Features

- **Advanced Preprocessing**: Uses CLAHE, Denoising, and Deskewing to clean 100-year-old degraded documents.
- **Deep Learning OCR**: Integrated `mindee/doctr` for robust script detection and recognition (supports multilingual/Indic).
- **Indic Script Support**: Hybrid fallback to Tesseract for specific languages (Hindi, Tamil, etc.).
- **Agentic AI**: Built-in AI Agent (OpenAI/LLM) that:
    - Structures extracted data (JSON).
    - Translates to vernacular languages.
    - Answers queries about the document content ("What is the survey number?").

## 🛠 Project Structure

- `src/preprocessing`: Image restoration pipeline.
- `src/ocr`: Core Engine wrapping Doctr and Tesseract.
- `src/agent`: LLM integration for reasoning and translation.
- `src/ui`: Streamlit frontend.

## 🚀 How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: You may need to install Tesseract OCR separately on your system for fallback support.*
   `(brew install tesseract)`

2. **Setup API Keys**
   - Create a `.env` file or enter your OpenAI API Key in the UI sidebar.

3. **Start the App**
   ```bash
   chmod +x run.sh
   ./run.sh
   # OR
   streamlit run src/main.py
   ```

## 🧩 Usage

1. Upload a scanned image of a land record or manuscript.
2. Click **Process Document**.
3. View the **Enhanced Image** (Preprocessing result).
4. See **Extracted Text** and **Structured JSON**.
5. Use the **Chat Interface** to ask questions in Hindi, English, etc.
