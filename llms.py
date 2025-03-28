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


load_dotenv()


llm = Ollama(model="mistral")

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
    
    # Extraction du texte PDF
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    
    cl.user_session.set("full_pdf_text", pdf_text)
    
    # Découpage du texte
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=50
    )
    texts = text_splitter.split_text(pdf_text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    
    # Création de la base vectorielle
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    # Configuration de la mémoire
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    # Création de la chaîne conversationnelle
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    
    # Définition des actions
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
    res = await chain.acall({"question": "Génère des exemples concrets basés sur le document"}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["answer"]).send()

@cl.action_callback("generate_quiz")
async def generate_quiz(action):
    chain = cl.user_session.get("chain")
    res = await chain.acall({"question": "Crée un quiz de 5 questions sur le contenu du document"}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["answer"]).send()

@cl.action_callback("generate_questions")
async def generate_questions(action):
    chain = cl.user_session.get("chain")
    res = await chain.acall({"question": "Propose 5 questions importantes à se poser sur ce document"}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["answer"]).send()

@cl.action_callback("generate_explanation")
async def generate_explanation(action):
    chain = cl.user_session.get("chain")
    res = await chain.acall(
        {"question": "Fournis une explication détaillée et pédagogique de ce concept en t'appuyant sur le document"},
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    answer = res["answer"]
    sources = "\n\nSources:\n" + "\n".join([f"- {doc.metadata['source']}" for doc in res["source_documents"]])
    await cl.Message(content=answer + sources).send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    sources = res["source_documents"]
    
    if sources:
        sources_content = "\n\nSources:\n" + "\n".join([f"- {doc.metadata['source']}" for doc in sources])
    else:
        sources_content = ""
    
    await cl.Message(content=answer + sources_content).send()

if __name__ == "__main__":
    cl.run()