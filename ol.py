import chainlit as cl
import ollama

model = "mistral"

@cl.on_message
async def handle_message(message: cl.Message):
    user_message = message.content  
    print(user_message)

    if user_message.lower() == "exit":
        await cl.Message(content="Fin de la session.").send()
        return

   
    response = ollama.chat(model=model, messages=[{"role": "user", "content": user_message}])
    answer = response['message']['content']
    

    await cl.Message(content=answer).send()

if __name__ == "__main__":
    cl.run(port=5000)  
