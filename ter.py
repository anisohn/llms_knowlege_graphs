import random
import chainlit as cl
from neo4j import GraphDatabase
from langchain_community.llms import Ollama

# Connexion à Neo4j
uri = "bolt://localhost:7689"
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
llm = Ollama(model="mistral")

# ---------------------------
# Fonctions base de données
# ---------------------------
def get_random_concepts(tx):
    query = """
    MATCH (c:Concept)
    WHERE c.name IS NOT NULL
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

# ---------------------------
# Chat start
# ---------------------------
@cl.on_chat_start
async def on_chat_start():
    with driver.session() as session:
        concepts = session.read_transaction(get_random_concepts)

    if not concepts:
        await cl.Message("Aucun concept trouvé dans la base de données.").send()
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
        await cl.Message("🎉 Fin du test ! Merci pour ta participation.").send()
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
    await cl.Message("Parfait ! ✅ Passons au suivant.").send()
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
        f"Explique les prérequis suivants pour bien comprendre le concept '{concept}' : {', '.join(prerequisites)}.\n"
        if prerequisites else "Ce concept ne nécessite aucun prérequis particulier.\n"
    )
    concept_text = f"Explique ensuite le concept '{concept}' avec des exemples concrets et des analogies."

    quiz_prompt = (
        f"Génère 3 questions Vrai/Faux avec les bonnes réponses, au format JSON comme ceci :\n"
        f"[{{'question': '...', 'answer': 'Vrai'}}, ...] sur le concept '{concept}'."
    )

    try:
        explanation = await cl.make_async(llm.invoke)(prereq_text + "\n\n" + concept_text)
        quiz_json = await cl.make_async(llm.invoke)(quiz_prompt)
    except Exception as e:
        await cl.Message(f"⚠️ Erreur pendant la génération : {e}").send()
        cl.user_session.set("current_index", cl.user_session.get("current_index") + 1)
        await ask_concept_question()
        return

    # Afficher explication
    await cl.Message(f"📘 **Explication de {concept}**\n\n{explanation.strip()}").send()

    try:
        # Convertir en liste Python (sécurisé)
        quiz_data = eval(quiz_json.strip())  # ou json.loads() si proprement formaté
        cl.user_session.set("current_quiz", quiz_data)
        cl.user_session.set("quiz_index", 0)
        cl.user_session.set("quiz_score", 0)

        await send_quiz_question()
    except Exception as e:
        await cl.Message("⚠️ Erreur dans la génération du quiz.").send()
        cl.user_session.set("current_index", cl.user_session.get("current_index") + 1)
        await ask_concept_question()

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

        # Prochain concept
        cl.user_session.set("current_index", cl.user_session.get("current_index") + 1)
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
# Lancer l'app
# ---------------------------
if __name__ == "__main__":
    cl.run()