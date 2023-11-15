import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to process the input and generate a response
def generate_response(question):
    # Tokenize the input
    input_ids = tokenizer.encode(question, add_special_tokens=True, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Example usage
question = "What are the symptoms of CoVID-19?"
response = generate_response(question)
print("Chatbot:", response)
