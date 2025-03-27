import ollama

model = "mistral" 

print("🤖 Modèle chargé. Posez une question (tapez 'exit' pour quitter).")

while True:
    question = input("\nVous : ")
    if question.lower() == "exit":
        print("Fin de la session.")
        break
    
  
    response = ollama.chat(model=model, messages=[{"role": "user", "content": question}])
    print("\nRéponse :", response['message']['content'])
