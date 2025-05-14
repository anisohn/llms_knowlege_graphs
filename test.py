import random
from neo4j import GraphDatabase
from langchain_community.llms import Ollama
import ollama
import PyPDF2
import chainlit as cl
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import random
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import json
chatCount = 0

# Connexion à Neo4j
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"



try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        
        result = session.run("RETURN 1 AS test")
        for record in result:
            print("Connexion réussie:", record["test"])
except Exception as e:
    print("Erreur de connexion à Neo4j:", e)

# Initialisation du modèle LLM
load_dotenv()

from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="mistral")

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
    
    full_text = pdf_text + "\n" + ocr_text
    # Si le texte extrait est vide, retournez une indication
    if not full_text.strip():
        return None  # Indique que le PDF ne contient pas de texte
    return full_text


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
    global chatCount 
    chatCount += 1
    if chatCount > 5:
        print("Limite de messages atteinte.")
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    
    res = await chain.acall({"question": message.content}, callbacks=[cb])
    answer = res["answer"]
    sources = res["source_documents"]
    
    sources_content = "\n\nSources:\n" + "\n".join([f"- {doc.metadata['source']}" for doc in sources]) if sources else ""
    await cl.Message(content=answer + sources_content).send()


# ---------------------------
# Fonctions base de données
# ---------------------------
def get_random_concepts(tx):
    query = """
    MATCH (c:Concept)
    WHERE c.name IS NOT NULL AND c.known = 0
    RETURN c.name AS concept
    ORDER BY rand()
    LIMIT 5
    """
    result = tx.run(query)
    return [record["concept"] for record in result]

def get_prerequisites(tx, concept):
    query = """
    MATCH (c:Concept)-[:PREREQUISITE]->(p:Concept)
    WHERE c.name = $concept
    RETURN p.name AS prerequisite
    """
    result = tx.run(query, concept=concept)
    return [record["prerequisite"] for record in result]

def get_all_concepts(tx):
    query = """
    MATCH (c:Concept)
    WHERE c.name IS NOT NULL 
    RETURN c.name AS concept
    ORDER BY rand()
    """
    result = tx.run(query)
    return [record["concept"] for record in result]

# ---------------------------
# Chat start
# ---------------------------
@cl.on_chat_start
async def on_chat_start():
    with driver.session() as session:
        concepts = session.read_transaction(get_random_concepts)

    if not concepts:
        await cl.Message("Aucun concept trouvé dans la base de données.").send()
        await handle_pdf_upload()
        return

    cl.user_session.set("concepts", concepts)
    cl.user_session.set("current_index", 0)

    await ask_concept_question()



# ---------------------------
# Étape 1 : Demande initiale
# ---------------------------

async def ask_concept_question():
    concepts = cl.user_session.get("concepts")
    index = cl.user_session.get("current_index")

    if index >= len(concepts):
        await cl.Message("🎓 Tu as terminé tous les concepts ! Place maintenant à l'analyse du PDF.").send()  
        await handle_pdf_upload()
        return

    concept = concepts[index]
    await cl.Message(
        content=f"Connais-tu le concept suivant : **{concept}** ? (Oui/Non)",
        actions=[
            cl.Action(name="yes", label="Oui", payload={"known": True}),
            cl.Action(name="no", label="Non", payload={"known": False})
        ]
    ).send()

# ---------------------------
# Si connu
# ---------------------------
@cl.action_callback("yes")
async def handle_known_concept(action):
    # Récupérer le concept actuel
    concepts = cl.user_session.get("concepts")
    index = cl.user_session.get("current_index")
    concept = concepts[index]

    # Mise à jour de la propriété 'known' à 1 dans Neo4j
    try:
        with driver.session() as session:
            session.run("""
                MATCH (c:Concept)
                WHERE c.name = $concept
                SET c.known = 1
            """, concept=concept)
        await cl.Message("Parfait ! ✅ Passons au suivant.").send()
    except Exception as e:
        await cl.Message(f"⚠️ Erreur lors de la mise à jour du concept : {e}").send()

    # Passer au concept suivant
    cl.user_session.set("current_index", cl.user_session.get("current_index") + 1)
    
    await ask_concept_question()

# ---------------------------
# Si inconnu → Explication + Quiz
# ---------------------------
@cl.action_callback("no")
async def handle_unknown_concept(action):
    concepts = cl.user_session.get("concepts")
    index = cl.user_session.get("current_index")
    concept = concepts[index]

    with driver.session() as session:
        prerequisites = session.read_transaction(get_prerequisites, concept)

    prereq_text = (
        f"""
        Pour bien comprendre le concept « {concept} », il est essentiel de maîtriser certains prérequis étroitement liés à ce sujet. 
        Voici les prérequis pertinents : {', '.join(prerequisites)}.
        Pour chaque prérequis identifié :
        - Présente uniquement les aspects directement utiles pour comprendre le concept « {concept} ».
        - Donne un exemple concret ou une analogie en lien direct avec le concept pour illustrer ces aspects.
        - Concentre-toi sur les notions clés permettant de faire le lien entre ce prérequis et le concept.
        - Propose une ressource ciblée (article, tutoriel ou vidéo) permettant d’approfondir spécifiquement ce prérequis dans le contexte du concept « {concept} ».
        """
        if prerequisites
        else "Ce concept ne nécessite aucun prérequis particulier.\n")


    concept_text = f"""
    Explique maintenant le concept '{concept}' de manière claire, ciblée et pédagogique :
    
    - Décris les idées clés du concept : à quoi il sert, pourquoi il est important, et dans quels contextes on l’utilise.
    - Détaille uniquement les usages les plus pertinents pour un étudiant débutant ou en difficulté.
    - Utilise des exemples ou analogies en lien avec les prérequis mentionnés précédemment.
    - Montre les erreurs fréquentes ou confusions possibles à éviter.
    - Suggère 1 ou 2 ressources bien choisies pour renforcer la compréhension et pratiquer le concept efficacement."""

    quiz_prompt = f"""
    Génère 3 questions Vrai/Faux au FORMAT JSON STRICT pour '{concept}'.
    Respecte scrupuleusement :
    - Guillemets doubles uniquement
    - Pas de texte hors JSON
    - Réponses uniquement 'Vrai'/'Faux'
    - Questions courtes (<20 mots)
    
    Exemple VALIDE :
    {{
        "quiz": [
            {{
                "question": "Le HTTP utilise le port 80 par défaut",
                "answer": "Vrai"
            }},
            {{
                "question": "SSL et TLS désignent le même protocole",
                "answer": "Faux"
            }}
        ]
    }}"""

    try:
        # Génération des contenus
        explanation = await cl.make_async(llm.invoke)(prereq_text + "\n\n" + concept_text)
        quiz_response = await cl.make_async(llm.invoke)(quiz_prompt)

        # Nettoyage du JSON
        quiz_json = quiz_response.strip()
        quiz_json = quiz_json.replace("“", '"').replace("”", '"') 
        
        # Extraction du JSON depuis les blocs Markdown
        if '```json' in quiz_json:
            quiz_json = quiz_json.split('```json')[1].split('```')[0]
        
        # Correction automatique des virgules manquantes
        quiz_json = quiz_json.replace('}{', '},{').replace('}\n{', '},\n{')
        
        # Validation et parsing
        try:
            data = json.loads(quiz_json)
            quiz_data = data.get("quiz", [])
            
            if not isinstance(quiz_data, list):
                raise ValueError("Structure 'quiz' invalide")
                
            # Validation des questions
            for i, q in enumerate(quiz_data):
                if not isinstance(q, dict):
                    raise ValueError(f"Question {i+1} n'est pas un objet")
                if 'question' not in q or 'answer' not in q:
                    raise ValueError(f"Question {i+1} manque des champs requis")
                
                # Normalisation des réponses
                q['answer'] = q['answer'].strip().title()
                if q['answer'] not in ['VRAI', 'FAUX']:
                    q['answer'] = 'Vrai'  # Valeur par défaut sécurisée

        except Exception as e:
            # Fallback en cas d'erreur persistante
            quiz_data = [{
                "question": f"Question {i+1} (Erreur technique)",
                "answer": "Vrai"
            } for i in range(3)]
            
            await cl.Message(
                f"⚠️ Problème de formatage du quiz. Erreur : {str(e)}\n"
                f"Réponse brute du modèle :\n{quiz_response}"
            ).send()

        # Affichage avec payload corrigé
        await cl.Message(
            content=f"📘 **{concept}**\n{explanation.strip()}",
            actions=[
                cl.Action(name="generate_quiz", label="🧠 Générer un nouveau quiz", payload={"concept": concept}),
                cl.Action(name="more_examples", label="💡 Plus d'exemples", payload={"concept": concept})
            ]
        ).send()

        # Sauvegarde des données
        cl.user_session.set("current_quiz", quiz_data)
        cl.user_session.set("quiz_index", 0)
        cl.user_session.set("quiz_score", 0)
        
        await send_quiz_question()

    except Exception as e:
        await cl.Message(
            f"⚠️ Erreur lors de la génération : {str(e)}\n"
            "Nous passons au concept suivant."
        ).send()
        cl.user_session.set("current_index", index + 1)
        await ask_concept_question()@cl.action_callback("no")


# ---------------------------
# Envoyer une question du quiz
# ---------------------------
async def send_quiz_question():
    quiz = cl.user_session.get("current_quiz")
    index = cl.user_session.get("quiz_index")

    if index >= len(quiz):
        score = cl.user_session.get("quiz_score")
        total = len(quiz)

        result_msg = f"✅ Tu as bien compris ! Score : {score}/{total}" if score == total else f"📘 Tu as eu {score}/{total}. Continue de réviser !"
        await cl.Message(result_msg).send()

        # Si toutes les questions sont répondues correctement, mettre à jour le concept comme "connu"
        if score == total:
            concepts = cl.user_session.get("concepts")
            current_index = cl.user_session.get("current_index")
            if current_index < len(concepts):
                concept = concepts[current_index]
                try:
                    with driver.session() as session:
                        session.run("""
                            MATCH (c:Concept)
                            WHERE c.name = $concept
                            SET c.known = 1
                        """, concept=concept)
                    await cl.Message(f"🎓 Le concept '{concept}' est maintenant marqué comme connu.").send()
                except Exception as e:
                    await cl.Message(f"⚠️ Erreur lors de la mise à jour du concept : {e}").send()

        # Passer au concept suivant
        cl.user_session.set("current_index", cl.user_session.get("current_index") + 1)

        # Vérifier si tous les concepts ont été vus
        concepts = cl.user_session.get("concepts")
        current_index = cl.user_session.get("current_index")

        if current_index >= len(concepts):
            await cl.Message("🎓 Tu as terminé tous les concepts ! Place maintenant à l'analyse du PDF.").send()
            await handle_pdf_upload()  # Lance l'analyse du PDF
        else:
            await ask_concept_question()
        return

    question = quiz[index]["question"]
    await cl.Message(
        content=f"❓ **Question {index+1} :** {question}",
        actions=[
            cl.Action(name="true", label="Vrai", payload={"response": "Vrai"}),
            cl.Action(name="false", label="Faux", payload={"response": "Faux"})
        ]
    ).send()

# ---------------------------
# Réponse au quiz
# ---------------------------
@cl.action_callback("true")
async def answer_true(action):
    await handle_quiz_response("Vrai")

@cl.action_callback("false")
async def answer_false(action):
    await handle_quiz_response("Faux")

async def handle_quiz_response(user_answer):
    quiz = cl.user_session.get("current_quiz")
    index = cl.user_session.get("quiz_index")
    correct = quiz[index]["answer"]

    if user_answer.lower() == correct.lower():
        await cl.Message("✅ Bonne réponse !").send()
        cl.user_session.set("quiz_score", cl.user_session.get("quiz_score") + 1)
    else:
        await cl.Message(f"❌ Mauvaise réponse. La bonne réponse était **{correct}**.").send()

    cl.user_session.set("quiz_index", index + 1)
    await send_quiz_question()

# ---------------------------
# Lancer l'analyse du PDF après la fin des quiz
# ---------------------------
async def handle_pdf_upload():
  
    
    files = None
    
    # Demande d'upload de fichier
    while files is None:
        files = await cl.AskFileMessage(
            content="📄 Veuillez uploader un fichier PDF pour commencer !",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180
        ).send()

    file = files[0]
    pdf_text = extract_text_from_pdf(file.path)

    if not pdf_text:
        await cl.Message(content="⚠️ Le PDF téléchargé ne contient pas de texte exploitable. Veuillez essayer avec un autre fichier.").send()
        return

    # Découpage du texte
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Embedding & indexation
    persist_dir = "./chroma_db"
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, 
        embeddings, 
        metadatas=metadatas,
        persist_directory=persist_dir ) # Ajout crucial
    docsearch.persist()

    # Mémoire de conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=ChatMessageHistory(),
        return_messages=True,
    )

    # Création de la chaîne de QA
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    cl.user_session.set("chain", chain)

    # Message d’attente pour l’analyse du PDF
    wait_message = await cl.Message("🔍 Analyse du PDF en cours...").send()

    # Résumé du contenu
    summary_prompt = f"""
    Fais un résumé structuré et concis du texte suivant, en mettant en valeur les concepts clés et idées principales :
    {pdf_text}
    """
    summary = await cl.make_async(llm.invoke)(summary_prompt)

    # Récupération de tous les concepts
    with driver.session() as session:
        concepts = session.read_transaction(get_all_concepts)
    cl.user_session.set("concepts", concepts)

    matched_concepts = set()

    # Détection des concepts mentionnés dans le résumé
    for concept in concepts:
        prompt = f"""
        Analyse ce résumé et détermine s'il mentionne ou traite du concept \"{concept}\".
        Réponds uniquement par 'Oui' ou 'Non'.

        Résumé :
        {summary}
        """
        try:
            response = await cl.make_async(llm.invoke)(prompt)
            if "oui" in response.strip().lower():
                matched_concepts.add(concept)

                with driver.session() as session:
                    result = session.run("MATCH (c:Concept) WHERE c.name = $concept RETURN c.known AS known", concept=concept)
                    known = result.single()["known"]

                explanation_prompt = f"""
                Explique le concept '{concept}' de manière claire et accessible :
                - Précise à quoi il sert et pourquoi il est important.
                - Utilise des exemples concrets et des analogies.
                - Propose des ressources pour approfondir.
                """

                if known == 0:
                    with driver.session() as session:
                        prerequisites = session.read_transaction(get_prerequisites, concept)

                    if prerequisites:
                        
                        prereq_text = ( f"""
                                       Pour bien comprendre le concept « {concept} », il est essentiel de maîtriser certains prérequis étroitement liés à ce sujet.
                                       Voici les prérequis pertinents : {', '.join(prerequisites)}.
                                       Pour chaque prérequis identifié :
                                       - Présente uniquement les aspects directement utiles pour comprendre le concept « {concept} ».
                                       - Donne un exemple concret ou une analogie en lien direct avec le concept pour illustrer ces aspects.
                                       - Concentre-toi sur les notions clés permettant de faire le lien entre ce prérequis et le concept.
                                       - Propose une ressource ciblée (article, tutoriel ou vidéo) permettant d’approfondir spécifiquement ce prérequis dans le contexte du concept « {concept} ».
                                       """
                                       if prerequisites
                                       else "Ce concept ne nécessite aucun prérequis particulier.\n")

                    else:
                        prereq_text = "Ce concept ne nécessite aucun prérequis particulier.\n"

                    explanation_text = prereq_text + "\n\n" + explanation_prompt
                elif known == 1 :
                    explanation_text = explanation_prompt

                explanation = await cl.make_async(llm.invoke)(explanation_text)
                await cl.Message(f"📘 **Explication de {concept}**\n\n{explanation.strip()}").send()

        except Exception as e:
            print(f"❌ Erreur lors de la vérification du concept '{concept}' : {e}")

    await wait_message.remove()

    if matched_concepts:
        cl.user_session.set("matched_concepts", matched_concepts)

        # Génération du quiz
        quiz_prompt = f"""
        Génère 5 questions de quiz à choix multiples (QCM), chacune portant sur un des concepts suivants : {', '.join(matched_concepts)}.
        Pour chaque question :
        - Propose une bonne réponse et trois distracteurs plausibles.
        - N'indique PAS la bonne réponse.
        - Utilise un format clair comme :
        **Question 1 :** Quel est le rôle de XYZ ?
        A. Réponse plausible
        B. Réponse plausible
        C. Réponse plausible
        D. Réponse plausible
        Adapte la langue du quiz à celle des concepts si besoin.
        """

        quiz_output = await cl.make_async(llm.invoke)(quiz_prompt)
        await cl.Message(f"🧠 **Quiz basé sur les concepts détectés :**\n\n{quiz_output.strip()}").send()
    else:
        await cl.Message("❌ Aucun concept détecté dans ce document.").send()

    cl.user_session.set("full_pdf_text", pdf_text)
    
    await cl.Message("🎓 Le PDF a été traité. Vous pouvez maintenant poser vos questions !").send()

    
              

  
   

   