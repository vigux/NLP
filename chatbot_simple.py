def chatbot_response(user_input):
    responses = {
        "hello": "Hello! How can I assist you today?",
        "hi": "Hi there! What can I do for you?",
        "how are you?": "I'm just a chatbot, but I'm here to help you!",
        "what is your name?": "I am ChatBot, your virtual assistant.",
        "bye": "Goodbye! Have a great day!"
    }

    user_input = user_input.lower()
    return responses.get(user_input, "I'm sorry, I don't understand that. Can you please rephrase?")

# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("ChatBot: Goodbye! Have a great day!")
        break
    response = chatbot_response(user_input)
    print(f"ChatBot: {response}")