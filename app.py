import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)



st.title("GPT-2 Text Generator")

prompt = st.text_input("Enter your prompt:")
max_length = st.slider("Max length", 10, 100, 50)

if st.button("Generate"):
    inputs = tokenizer(prompt, return_tensors="pt")
    attention_mask = inputs['attention_mask']
    pad_token_id = tokenizer.pad_token_id
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    st.write(generated_text)