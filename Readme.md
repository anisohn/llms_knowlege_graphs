```markdown
# 📚 Apprentissage Personnalisé avec LLMs : Combinaison de Graphes de Connaissances et Datasets  

Ce projet vise à exploiter la puissance des **graphes de connaissances** et des **datasets d'apprentissage** pour créer un modèle de **LLM personnalisé**, capable de générer des explications adaptées en fonction du niveau et des lacunes d'un élève.  

## 🚀 Objectif

Créer un système interactif d'apprentissage qui :
- Analyse des documents PDF pour en extraire les connaissances
- Génère des interactions pédagogiques personnalisées
- Maintient un contexte conversationnel avec mémoire
- Produit des exercices et quiz adaptatifs

## ✨ Fonctionnalités Clés

- **Analyse de documents PDF** avec extraction de texte et découpage intelligent
- **Chat interactif** avec mémoire de conversation contextuelle
- **Génération automatique** :
  - Exemples concrets basés sur le document
  - Quiz personnalisés avec questions variées
  - Suggestions de questions d'approfondissement
- **Recherche sémantique** dans les documents avec ChromaDB
- **Intégration de modèles LLM locaux** via Ollama

## 🛠 Technologies Utilisées

- **Ollama** : Interface locale pour l'exécution de LLMs (Mistral)
- **LangChain** : Orchestration des flux de traitement NLP
- **ChromaDB** : Base de données vectorielle pour la recherche sémantique
- **Chainlit** : Interface utilisateur conversationnelle
- **PyPDF2** : Extraction de contenu depuis fichiers PDF
- **RecursiveCharacterTextSplitter** : Découpage contextuel du texte

## 📦 Installation

1. Prérequis :
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

2. Installer les dépendances :
```bash
pip install chainlit langchain langchain-community langchain-chroma pypdf2 python-dotenv ollama
```

## 🖥 Utilisation

1. Démarrer l'application :
```bash
chainlit run votre_fichier.py -w
```

2. Workflow :
- Upload d'un document PDF
- Interaction naturelle via :
  - Questions directes sur le contenu
  - Boutons d'actions prédéfinies :
    - Générer des exemples pratiques
    - Créer un quiz interactif
    - Obtenir des questions d'approfondissement
- Réponses contextuelles avec sources documentaires

## ⚙️ Customisation

Modifier dans le code :
```python
# Modèle LLM
llm = Ollama(model="mistral")  # Changer le modèle

# Paramètres de découpage
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # Taille des segments textuels
    chunk_overlap=50   # Chevauchement contextuel
)

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Modèle d'embedding
```

## Solutions pour traiter les images dans un PDF 

- 1 : pip install pytesseract pdf2image pillow


## ⚠️ Important

L'application nécessite :
- Ollama installé localement
- Les modèles Mistral et nomic-embed-text téléchargés
- Une connexion internet pour le premier lancement (téléchargement des modèles)

```
```
##  des prompts éducatifs

-  Cadrer strictement le domaine (éducation/contenu du PDF)
-  Expliciter le rôle du modèle et les attentes
-  Personnaliser le niveau de complexité
  
* Prompt  de Génération d'exemples :
 Générer 5 exemples concrets qui illustrent les concepts clés du document PDF. Ces exemples peuvent inclure des analogies     

* Prompt  de Génération de quiz :

  Créer 5 questions de quiz basées sur le texte, chaque question ayant :

- Une bonne réponse
- Trois alternatives plausibles mais incorrectes
- Une explication détaillée de la réponse correcte   

* Prompt  de Génération de questions :

 Générer 5 questions ouvertes qui encouragent l’analyse et la réflexion sur le texte. Les questions sont basées sur trois thèmes :

- Factual Details (Détails factuels) : Questions sur des faits spécifiques du document.
- Interpretative Insights (Interprétations) : Questions sur le sens et l’analyse des idées du texte.
- Critical Evaluations (Évaluations critiques) : Questions qui poussent à argumenter et critiquer le contenu.


* Prompt  de Génération d'explications :

 Fournir une explication claire et pédagogique d’un concept clé du texte,