import ollama
import PyPDF2
import chainlit as cl
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import pytesseract
from pdf2image import convert_from_path
import pandas as pd

load_dotenv()

llm = OllamaLLM(model="mistral")

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
            print(f"Pas de texte trouvé sur la page {page_number + 1}, tentative d'OCR.")
            img = images[page_number]
            ocr_text += pytesseract.image_to_string(img)
    
    full_text = pdf_text + "\n" + ocr_text
    return full_text if full_text.strip() else None

def initialize_known_nodes(csv_path):
    df = pd.read_csv(csv_path)
    if "known" not in df.columns:
        df["known"] = 0
    df.to_csv(csv_path, index=False)
    return df

def select_random_concepts(df, n=5):
    return df.sample(n)

@cl.on_chat_start
async def on_chat_start():
    csv_path = "full_graph_updated.csv"
    df = initialize_known_nodes(csv_path)
    selected_concepts = select_random_concepts(df)
    
    for _, row in selected_concepts.iterrows():
        concept = row["name"]
        msg = cl.AskUserMessage(
            content=f"Connaissez-vous le concept suivant : {concept} ? (oui/non)", 
            timeout=60
        )
        await msg.send()
        response = await msg.wait()
        
        if response['content'].lower() in ["oui", "yes"]:
            df.loc[df["name"] == concept, "known"] = 1
    
    df.to_csv(csv_path, index=False)
    await cl.Message(content="Merci pour vos réponses ! Veuillez uploader un fichier PDF.").send()
    
    files = None
    while not files:
        msg = cl.AskUserMessage(
            content="Uploader un PDF:",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180
        )
        await msg.send()
        response = await msg.wait()
        files = response.get("files")
    
    file = files[0]
    if pdf_text := extract_text_from_pdf(file.path):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        texts = text_splitter.split_text(pdf_text)
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        docsearch = await cl.make_async(Chroma.from_texts)(texts, embeddings)
        
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True
        )
        
        cl.user_session.set(
            "chain", 
            ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=docsearch.as_retriever(),
                memory=memory
            )
        )
        await cl.Message(content=f"PDF traité! Posez vos questions.").send()
    else:
        await cl.Message(content="Erreur: PDF vide ou illisible").send()

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    res = await chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    
    answer = res["answer"]
    if sources := [doc.metadata.get("source", "") for doc in res["source_documents"]]:
        answer += f"\n\nSources: {', '.join(sources)}"
    
    await cl.Message(content=answer).send()

# Pour les versions de Chainlit < 1.0, utiliser cette gestion des actions
@cl.action_callback("generate_examples")
async def on_action(action: cl.Action):
    chain = cl.user_session.get("chain")
    res = await chain.acall("Génère 3 exemples concrets", callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["answer"]).send()

@cl.action_callback("generate_quiz")
async def on_action(action: cl.Action):
    chain = cl.user_session.get("chain")
    res = await chain.acall("Crée un quiz de 5 questions", callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["answer"]).send()

if __name__ == "__main__":
    cl.run()