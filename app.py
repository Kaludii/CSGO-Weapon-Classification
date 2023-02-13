# Import the required Libraries
import gradio as gr
import numpy as np
import pandas as pd
import pickle
import transformers 
from transformers import AutoTokenizer, AutoConfig,AutoModelForSequenceClassification,TFAutoModelForSequenceClassification
from scipy.special import softmax
# Requirements
model_path = "Kaludi/Reviews-Sentiment-Analysis"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

# ---- Function to process the input and return prediction
def sentiment_analysis(text):
    text = preprocess(text)

    encoded_input = tokenizer(text, return_tensors = "pt") # for PyTorch-based models
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)
    
    # Format output dict of scores
    labels = ["Negative", "Positive"]
    scores = {l:float(s) for (l,s) in zip(labels, scores_) }
    
    return scores


# ---- Gradio app interface
app = gr.Interface(fn = sentiment_analysis,
                   inputs = gr.Textbox("Write your text or review here..."),
                   outputs = "label",
                   title = "Sentiment Analysis of Customer Reviews",
                   description  = "A tool that analyzes the overall sentiment of customer reviews for a specific product or service, whether it's positive or negative. This analysis is performed by using natural language processing algorithms and machine learning from the model 'Reviews-Sentiment-Analysis' trained by Kaludi, allowing businesses to gain valuable insights into customer satisfaction and improve their products and services accordingly.",
                   article = "<p style='text-align: center'><a href='https://github.com/Kaludii'>Github</a> | <a href='https://huggingface.co/Kaludi'>HuggingFace</a></p>",
                   interpretation = "default",
                   examples = [["I was extremely disappointed with this product. The quality was terrible and it broke after only a few days of use. Customer service was unhelpful and unresponsive. I would not recommend this product to anyone."],[ "I am so impressed with this product! The quality is outstanding and it has exceeded all of my expectations. The customer service team was also incredibly helpful and responsive to any questions I had. I highly recommend this product to anyone in need of a top-notch, reliable solution."],["I don't feel like you trust me to do my job."],["This service was honestly one of the best I've experienced, I'll definitely come back!"]]
                   )

app.launch()