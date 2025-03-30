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
## Prompts Éducatifs


### 2. Prompts de Génération d'Exemples
**Objectif :** Générer 5 exemples concrets qui illustrent les concepts clés du document PDF.

- Ces exemples peuvent inclure des analogies ou des situations réelles pour mieux comprendre les concepts.
  
**Exemple de prompt :**
_Générez 5 exemples pratiques illustrant les concepts clés du document PDF. Utilisez des analogies, des situations réelles ou des comparaisons pertinentes._

---

### 3. Prompts de Génération de Quiz
**Objectif :** Créer 5 questions de quiz basées sur le texte, avec des réponses et explications détaillées.

Chaque question doit comporter :
- Une bonne réponse.
- Trois alternatives plausibles mais incorrectes.
- Une explication détaillée de la réponse correcte.

**Exemple de prompt :**
_Créez 5 questions de quiz basées sur le texte, chaque question ayant :_
- _Une bonne réponse._
- _Trois alternatives plausibles mais incorrectes._
- _Une explication détaillée de la réponse correcte._

---

### 4. Prompts de Génération de Questions
**Objectif :** Générer 5 questions ouvertes qui encouragent l’analyse et la réflexion sur le texte. Les questions sont réparties en trois catégories :

- **Factual Details (Détails factuels)** : Questions sur des faits spécifiques du document.
- **Interpretative Insights (Interprétations)** : Questions sur le sens et l’analyse des idées du texte.
- **Critical Evaluations (Évaluations critiques)** : Questions qui poussent à argumenter et critiquer le contenu.

**Exemple de prompt :**
_Générez 5 questions ouvertes sur le texte, couvrant les trois thèmes suivants :_
- _Détails factuels._
- _Interprétation des idées._
- _Évaluation critique du contenu._

---

### 5. Prompts de Génération d'Explications
**Objectif :** Fournir une explication claire et pédagogique d’un concept clé du texte.

**Exemple de prompt :**
_Fournissez une explication détaillée et claire d’un concept important du texte, en utilisant un langage simple et des exemples pour faciliter la compréhension._

---
