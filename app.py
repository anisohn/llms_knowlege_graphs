from flask import Flask, request, jsonify
import ollama
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
from PIL import Image

load_dotenv()

app = Flask(__name__)

llm = Ollama(model="mistral")
sessions = {}

def extract_text_from_pdf(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    pdf_text = ""
    images = convert_from_path(file_path)
    ocr_text = ""
    
    for page_number, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            pdf_text += text
        else:
            img = images[page_number]
            ocr_text += pytesseract.image_to_string(img)
    
    full_text = pdf_text + "\n" + ocr_text
    return full_text if full_text.strip() else None

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    print("Fichier reçu:", file.filename)
    print("Chemin du fichier:", file_path)

    
    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    pdf_text = extract_text_from_pdf(file_path)
    if not pdf_text:
        return jsonify({"error": "PDF contains no readable text"}), 400
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))])
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=docsearch.as_retriever(), memory=memory, return_source_documents=True)
    
    session_id = str(random.randint(1000, 9999))
    sessions[session_id] = {"chain": chain, "text": pdf_text}
    
    return jsonify({"session_id": session_id, "message": "PDF processed successfully"})

@app.route('/action/<session_id>', methods=['POST'])
def perform_action(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Invalid session ID"}), 400
    
    action_type = request.json.get("type")
    text = sessions[session_id]["text"]
    chain = sessions[session_id]["chain"]
    
    if action_type == "generate_examples":
        prompt = f"Generate exactly 5 creative examples illustrating key concepts from the text:\n\n{text}"
    elif action_type == "generate_quiz":
        prompt = f"Generate 5 quiz questions with one correct answer, three incorrect options, and explanations:\n\n{text}"
    elif action_type == "generate_questions":
        prompt = f"Generate 5 unique questions based on the text:\n\n{text}"
    elif action_type == "generate_explanation":
        prompt = f"Provide a detailed pedagogical explanation of a key concept from the text:\n\n{text}"
    else:
        return jsonify({"error": "Invalid action type"}), 400
    
    res = chain.invoke({"question": prompt})
    return jsonify({"response": res["answer"]})

@app.route('/chat/<session_id>', methods=['POST'])
def chat(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Invalid session ID"}), 400
    
    user_question = request.json.get("question")
    chain = sessions[session_id]["chain"]
    
    res = chain.invoke({"question": user_question})
    return jsonify({"response": res["answer"]})

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)



