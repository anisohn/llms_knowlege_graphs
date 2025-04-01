from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os
import random
import pytesseract
from pdf2image import convert_from_path
import logging
import traceback

# Configuration initiale
load_dotenv()
app = Flask(__name__)
CORS(app)  # Activation CORS

# Configuration du logger
app.logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
app.logger.addHandler(handler)

# Variables globales
llm = Ollama(model="mistral")
sessions = {}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_pdf(file_path):
    """Extrait le texte d'un PDF avec fallback OCR"""
    try:
        pdf = PyPDF2.PdfReader(file_path)
        pdf_text = ""
        images = convert_from_path(file_path)
        ocr_text = ""

        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pdf_text += text
            else:
                if page_number < len(images):
                    img = images[page_number]
                    ocr_text += pytesseract.image_to_string(img)
                else:
                    app.logger.warning(f"Page {page_number+1} manquante pour OCR")

        full_text = pdf_text + "\n" + ocr_text
        return full_text.strip() if full_text.strip() else None

    except Exception as e:
        app.logger.error(f"Erreur extraction PDF: {str(e)}")
        return None

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Endpoint pour l'upload de PDF"""
    try:
        if 'file' not in request.files:
            app.logger.error("Aucun fichier dans la requête")
            return jsonify({"error": "Aucun fichier reçu"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nom de fichier vide"}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Seuls les PDF sont acceptés"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        app.logger.info(f"Fichier sauvegardé: {file_path}")

        if os.path.getsize(file_path) == 0:
            return jsonify({"error": "Fichier vide"}), 400

        pdf_text = extract_text_from_pdf(file_path)
        if not pdf_text:
            return jsonify({"error": "PDF illisible ou vide"}), 400

        # Traitement LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=50
        )
        texts = text_splitter.split_text(pdf_text)
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        docsearch = Chroma.from_texts(
            texts, 
            embeddings, 
            metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))]
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=docsearch.as_retriever(),
            memory=memory,
            return_source_documents=True
        )

        session_id = str(random.randint(1000, 9999))
        sessions[session_id] = {
            "chain": chain,
            "text": pdf_text,
            "file_path": file_path
        }

        return jsonify({
            "session_id": session_id,
            "message": "PDF traité avec succès",
            "pages": len(texts)
        })

    except Exception as e:
        app.logger.error(f"Erreur critique: {traceback.format_exc()}")
        return jsonify({"error": f"Erreur interne: {str(e)}"}), 500

@app.route('/action/<session_id>', methods=['POST'])
def perform_action(session_id):
    """Génération de contenu pédagogique"""
    try:
        if session_id not in sessions:
            return jsonify({"error": "Session invalide"}), 404

        data = request.get_json()
        if not data or 'type' not in data:
            return jsonify({"error": "Type d'action manquant"}), 400

        action_type = data['type']
        text = sessions[session_id]["text"]
        chain = sessions[session_id]["chain"]

        prompts = {
            "generate_examples": "Génère 5 exemples concrets illustrant les concepts clés de ce texte :\n\n",
            "generate_quiz": "Crée un quiz de 5 questions avec réponses et explications :\n\n",
            "generate_questions": "Formule 5 questions pertinentes sur ce contenu :\n\n",
            "generate_explanation": "Explique en détail ce concept clé :\n\n"
        }

        if action_type not in prompts:
            return jsonify({"error": "Action non supportée"}), 400

        response = chain.invoke({"question": prompts[action_type] + text})
        return jsonify({
            "response": response["answer"],
            "sources": [doc.metadata["source"] for doc in response["source_documents"]]
        })

    except Exception as e:
        app.logger.error(f"Erreur action: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat/<session_id>', methods=['POST'])
def chat(session_id):
    """Chat interactif avec le document"""
    try:
        if session_id not in sessions:
            return jsonify({"error": "Session invalide"}), 404

        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Question manquante"}), 400

        chain = sessions[session_id]["chain"]
        response = chain.invoke({"question": data['question']})

        return jsonify({
            "response": response["answer"],
            "sources": [doc.metadata["source"] for doc in response["source_documents"]]
        })

    except Exception as e:
        app.logger.error(f"Erreur chat: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({
        "status": "OK",
        "sessions_actives": len(sessions),
        "version": "1.0.0"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)