import gradio as gr

def echo(text):
    return text

gr.Interface(echo, "text", "text").launch()