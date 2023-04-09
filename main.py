from fastapi import FastAPI
from typing import Optional
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer


app = FastAPI()



# Load the pre-trained summarization model and tokenizer
model_name = "t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the summarization function
@app.post("/{text}")
def summarize_text(text):
    # Encode the input text and generate a summary using the model
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1024, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Concatenate the input text and summary into a single passage
    passage =  summary
    
    return {"output":passage}
