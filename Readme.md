# 📚 Apprentissage Personnalisé avec LLMs : Combinaison de Graphes de Connaissances et Datasets  

Ce projet vise à exploiter la puissance des **graphes de connaissances** et des **datasets d’apprentissage** pour créer un modèle de **LLM personnalisé**, capable de générer des explications adaptées en fonction du niveau et des lacunes d’un élève.  

## 🚀 Objectif  

L’objectif est de :  

- **Utiliser un dataset** (exemple : [MathQA](https://github.com/karan-13/MathQA) pour les maths, [EdNet](https://github.com/riiid/ednet) pour l’apprentissage en général).  
- **Créer un graphe de connaissances** qui structure les concepts et suit la progression de l’élève.  
- **Entraîner un modèle LLM** sur ce dataset et l’utiliser conjointement avec le graphe pour générer des réponses intelligentes et adaptées.  

## 🏗️ Architecture de la solution  

1️⃣ **Dataset** → Sert à entraîner un modèle LLM afin de comprendre et expliquer les concepts.  
2️⃣ **Graphe de connaissances** → Suit l’évolution de l’élève, identifie ses forces/faiblesses et guide l’adaptation du contenu.  
3️⃣ **LLM personnalisé** → Génère des explications adaptées en s’appuyant sur les informations du graphe et les connaissances du dataset.  


