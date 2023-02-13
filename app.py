import gradio as gr
from transformers import pipeline

examples = ["examples/example_0.jpg", 
            "examples/example_1.png", 
            "examples/example_2.png", 
            "examples/example_3.png", 
            "examples/example_4.jpg",
            "examples/example_5.png"]

pipe = pipeline(task="image-classification", 
                model="Kaludi/csgo-weapon-classification")
gr.Interface.from_pipeline(pipe, 
                           title="CSGO Weapon Image Classification",
                           description = "This is a CSGO Weapon Classifier Model that has been trained by <strong><a href='https://huggingface.co/Kaludi'>Kaludi</a></strong> to recognize <strong>11</strong> different types of Counter-Strike: Global Offensive (CSGO) Weapons, which include <strong>AK-47,AWP,Famas,Galil-AR,Glock,M4A1,M4A4,P-90,SG-553,UMP,USP</strong>. The model is capable of accurately classifying the weapon name present in an image. With its deep understanding of the characteristics of each weapon in the game, the model is a valuable tool for players and fans of CSGO.",
                           article = "<p style='text-align: center'><a href='https://github.com/Kaludii'>Github</a> | <a href='https://huggingface.co/Kaludi'>HuggingFace</a></p>",
                           examples=examples,
                           ).launch()