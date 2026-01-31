import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class HeritageAgent:
    def __init__(self, provider="openai", model="gpt-4o"):
        self.provider = provider
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Assumes OpenAI for now
        
        self.system_prompt = """
        You are an expert AI archivist and translator specializing in Indian Heritage Documents.
        Your task is to:
        1. Analyze OCR text from old land records, manuscripts, and legal documents.
        2. Correct likely OCR errors based on context (e.g., correcting 'Khuta' to 'Khata').
        3. Extract structured entities: Owner Name, Survey Number, Land Type, Date.
        4. Translate content into modern English or specific Indian languages upon request.
        5. Answer user queries based ONLY on the provided document text.
        
        If the text is unclear, state that rather than hallucinating.
        """

    def process_document(self, ocr_text, language_hint="English/Hindi"):
        """
        Initial pass to structure the document.
        """
        prompt = f"""
        Here is the raw OCR text extracted from a heritage document (Language: {language_hint}):
        
        '''
        {ocr_text}
        '''
        
        Please extract the following fields in JSON format:
        - DocumentType (e.g., Land Record, Sale Deed, Rent Agreement, Affidavit)
        - Date (Execution Date or relevant dates)
        - PrincipalEntities (List of Buyers, Sellers, Authorities)
        - KeyClauses (If it's an agreement, list key points)
        - LocationData (City, Village, Survey Numbers - if applicable)
        - Summary (Brief description of the visible content)
        
        If specific fields match "Land Record" (like Khata/Survey No), extract them.
        If it is a generic agreement, extract the Parties and Purpose.
        Output strictly valid JSON.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

    def chat(self, ocr_text, user_query, target_language="English"):
        """
        Chat with the document.
        """
        prompt = f"""
        Document Context:
        '''
        {ocr_text}
        '''
        
        User Query: {user_query}
        Target Output Language: {target_language}
        
        Answer the query based on the document. If the answer involves technical terms (Khata, Khasra), explain them briefly.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
