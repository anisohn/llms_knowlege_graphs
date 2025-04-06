import chainlit as cl
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
import random

from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama  # Ou ton modèle LLM préféré
from chainlit.input_widget import AskFileMessage

llm = Ollama(model="llama3")  # Utilise ton modèle LLM ici

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
    return None if not full_text.strip() else full_text

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="📄 Veuillez uploader un fichier PDF pour commencer !",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180
        ).send()

    file = files[0]
    pdf_text = extract_text_from_pdf(file.path)

    if pdf_text is None:
        await cl.Message(content="❌ Le PDF ne contient pas de texte détectable. Essayez un autre fichier.").send()
        return

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

    cl.user_session.set("chain", chain)

    await cl.Message(content=f"✅ Le fichier `{file.name}` a été traité avec succès !\n\n🔄 Génération automatique des contenus en cours...").send()

    await auto_generate_all(chain, pdf_text)

async def auto_generate_all(chain, text):
    seed = random.randint(1, 100)

    # Générer un quiz
    quiz_prompt = (
        f"Using seed {seed}, generate exactly 5 quiz questions. Each should have one correct answer, "
        f"three incorrect alternatives, and an explanation for the correct choice.\n\n{text}"
    )
    quiz_res = await chain.acall({"question": quiz_prompt}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content="📘 **Quiz généré :**\n\n" + quiz_res["answer"]).send()

    # Générer des exemples
    examples_prompt = f"Using seed {seed}, generate exactly 5 creative examples that illustrate key concepts from the text.\n\n{text}"
    examples_res = await chain.acall({"question": examples_prompt}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content="💡 **Exemples générés :**\n\n" + examples_res["answer"]).send()

    # Générer des questions de réflexion
    themes = ["Factual Details", "Interpretative Insights", "Critical Evaluations"]
    theme = random.choice(themes)
    questions_prompt = f"Using seed {seed}, generate 5 unique questions focusing on '{theme}'.\n\n{text}"
    questions_res = await chain.acall({"question": questions_prompt}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=f"❓ **Questions de réflexion ({theme}) :**\n\n" + questions_res["answer"]).send()

    # Générer une explication pédagogique
    explanation_prompt = "Provide a detailed and pedagogical explanation of a key concept from the text.\n\n" + text
    explanation_res = await chain.acall({"question": explanation_prompt}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    sources = "\n\n📎 **Sources :**\n" + "\n".join([f"- {doc.metadata['source']}" for doc in explanation_res["source_documents"]])
    await cl.Message(content="📚 **Explication pédagogique :**\n\n" + explanation_res["answer"] + sources).send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall({"question": message.content}, callbacks=[cb])
    answer = res["answer"]
    sources = res.get("source_documents", [])

    sources_content = "\n\n📎 **Sources :**\n" + "\n".join([f"- {doc.metadata['source']}" for doc in sources]) if sources else ""
    await cl.Message(content=answer + sources_content).send()
