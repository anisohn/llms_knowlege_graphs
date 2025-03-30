import ollama
import PyPDF2
import chainlit as cl
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os
import random
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

load_dotenv()

llm = Ollama(model="mistral")

def extract_text_from_pdf(file_path):
    """
    Extrait le texte du fichier PDF. Si le PDF contient uniquement des images, l'OCR est utilisé pour extraire le texte des images.
    Si le PDF contient à la fois du texte et des images, le texte est extrait et l'OCR est appliqué aux pages sans texte.
    
    :param file_path: Chemin vers le fichier PDF.
    :return: Le texte extrait du PDF.
    """
    pdf = PyPDF2.PdfReader(file_path)
    pdf_text = ""
    images = convert_from_path(file_path)
    ocr_text = ""
    
    for page_number, page in enumerate(pdf.pages):
        # Tenter d'extraire le texte
        text = page.extract_text()
        if text:
            pdf_text += text
        else:
            # Si aucun texte n'est extrait, appliquer l'OCR sur l'image de cette page
            print(f"Pas de texte trouvé sur la page {page_number + 1}, tentative d'OCR.")
            img = images[page_number]
            ocr_text += pytesseract.image_to_string(img)
    
    # Retourner à la fois le texte extrait et le texte OCR si nécessaire
    return pdf_text + "\n" + ocr_text

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Veuillez uploader un fichier PDF pour commencer!",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180
        ).send()
    
    file = files[0]
    pdf_text = extract_text_from_pdf(file.path)
    cl.user_session.set("full_pdf_text", pdf_text)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=50
    )
    texts = text_splitter.split_text(pdf_text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    
    actions = [
        cl.Action(name="generate_examples", label="Générer des exemples", payload={"type": "generate_examples"}),
        cl.Action(name="generate_quiz", label="Générer un quiz", payload={"type": "generate_quiz"}),
        cl.Action(name="generate_questions", label="Générer des questions", payload={"type": "generate_questions"}),
        cl.Action(name="generate_explanation", label="Générer une explication", payload={"type": "generate_explanation"})
    ]
    
    await cl.Message(
        content=f"Traitement de `{file.name}` terminé! Posez vos questions ou utilisez les boutons ci-dessous.",
        actions=actions
    ).send()
    
    cl.user_session.set("chain", chain)

@cl.action_callback("generate_examples")
async def generate_examples(action):
    chain = cl.user_session.get("chain")
    random_seed = random.randint(1, 100)
    text = cl.user_session.get("full_pdf_text")
    prompt = (f"Using seed {random_seed}, generate exactly 5 creative examples that illustrate key concepts from the text.\n\n{text}")
    res = await chain.acall({"question": prompt}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["answer"]).send()

@cl.action_callback("generate_quiz")
async def generate_quiz(action):
    chain = cl.user_session.get("chain")
    random_seed = random.randint(1, 100)
    text = cl.user_session.get("full_pdf_text")
    prompt = (f"Using seed {random_seed}, generate exactly 5 quiz questions. Each should have one correct answer, "
              f"three incorrect alternatives, and an explanation for the correct choice.\n\n{text}")
    res = await chain.acall({"question": prompt}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["answer"]).send()

@cl.action_callback("generate_questions")
async def generate_questions(action):
    chain = cl.user_session.get("chain")
    themes = ["Factual Details", "Interpretative Insights", "Critical Evaluations"]
    random_theme = random.choice(themes)
    random_seed = random.randint(1, 100)
    text = cl.user_session.get("full_pdf_text")
    prompt = (f"Using seed {random_seed}, generate 5 unique questions focusing on '{random_theme}'.\n\n{text}")
    res = await chain.acall({"question": prompt}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["answer"]).send()

@cl.action_callback("generate_explanation")
async def generate_explanation(action):
    chain = cl.user_session.get("chain")
    text = cl.user_session.get("full_pdf_text")
    prompt = ("Provide a detailed and pedagogical explanation of a key concept from the text.\n\n" + text)
    res = await chain.acall({"question": prompt}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    answer = res["answer"]
    sources = "\n\nSources:\n" + "\n".join([f"- {doc.metadata['source']}" for doc in res["source_documents"]])
    await cl.Message(content=answer + sources).send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    
    res = await chain.acall({"question": message.content}, callbacks=[cb])
    answer = res["answer"]
    sources = res["source_documents"]
    
    sources_content = "\n\nSources:\n" + "\n".join([f"- {doc.metadata['source']}" for doc in sources]) if sources else ""
    await cl.Message(content=answer + sources_content).send()

if __name__ == "__main__":
    cl.run()
