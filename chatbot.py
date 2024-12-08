#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:15:13 2024

@author: abhigyanbhowal
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
def get_bot_response(user_input):
    # Format the conversation
    input_text = f"User: {user_input}\nBot:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate response
    output = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract bot's response
    bot_response = response.split("Bot:")[-1].strip()
    return bot_response
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Welcome to the Chatbot API! Use the /chat endpoint to interact with the chatbot.", 200

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    if not user_input:
        return jsonify({"error": "Please provide user_input in JSON format"}), 400
    
    bot_response = get_bot_response(user_input)  # Assuming get_bot_response is defined
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from datasets import load_dataset

dataset = load_dataset("json", data_files="data.json")
from transformers import Trainer, TrainingArguments

# Tokenize dataset
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_data = dataset.map(tokenize, batched=True)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
)
trainer.train()


