import random
import chainlit as cl
from neo4j import GraphDatabase
from langchain_community.llms import Ollama

# Connexion à Neo4j
uri = "bolt://localhost:7689"  # Port par défaut
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

# Fonction pour récupérer 5 concepts aléatoires
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

# Fonction pour récupérer les prérequis d'un concept
def get_prerequisites(tx, concept):
    query = """
    MATCH (c:Concept)-[:PREREQUISITE]->(p:Concept)
    WHERE c.name = $concept
    RETURN p.name AS prerequisite
    """
    result = tx.run(query, concept=concept)
    return [record["prerequisite"] for record in result]

@cl.on_chat_start
async def on_chat_start():
    with driver.session() as session:
        concepts = session.read_transaction(get_random_concepts)

    if not concepts:
        await cl.Message(content="Aucun concept trouvé dans la base de données.").send()
        return

    cl.user_session.set("concepts", concepts)
    cl.user_session.set("current_index", 0)

    await ask_concept_question()

async def ask_concept_question():
    concepts = cl.user_session.get("concepts")
    index = cl.user_session.get("current_index")

    if index >= len(concepts):
        await cl.Message(content="🎉 Fin du test !").send()
        return

    concept = concepts[index]
    await cl.Message(
        content=f"Connais-tu le concept suivant : **{concept}** ? (Oui/Non)",
        actions=[
            cl.Action(name="yes", label="Oui", payload={"known": True}),
            cl.Action(name="no", label="Non", payload={"known": False})
        ]
    ).send()

@cl.action_callback("yes")
async def handle_known_concept(action):
    await cl.Message(content="Bien ! ✅ Passons au suivant.").send()
    cl.user_session.set("current_index", cl.user_session.get("current_index") + 1)
    await ask_concept_question()

@cl.action_callback("no")
async def handle_unknown_concept(action):
    concepts = cl.user_session.get("concepts")
    index = cl.user_session.get("current_index")
    concept = concepts[index]

    with driver.session() as session:
        prerequisites = session.read_transaction(get_prerequisites, concept)

    explanation_prompt = (
        f"Explique le concept de '{concept}' de manière détaillée pour un débutant. "
        f"Prérequis nécessaires: {', '.join(prerequisites) if prerequisites else 'aucun'}. "
        f"Donne des analogies et des exemples concrets."
    )

    response = await cl.make_async(llm.invoke)(explanation_prompt)

    if not response.strip():
        response = "Désolé, je n'ai pas pu générer d'explication pour ce concept."

    await cl.Message(content=f"📚 **{concept}** :\n\n{response}").send()

    cl.user_session.set("current_index", cl.user_session.get("current_index") + 1)
    await ask_concept_question()

if __name__ == "__main__":
    cl.run()
