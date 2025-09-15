import os
from huggingface_hub import InferenceClient

## You need a token from https://hf.co/settings/tokens, ensure that you select 'read' as the token type. If you run this on Google Colab, you can set it up in the "settings" tab under "secrets". Make sure to call it "HF_TOKEN"
# HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(model="meta-llama/Llama-4-Scout-17B-16E-Instruct")
# HF_TOKEN = os.environ.get("HF_TOKEN")
output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of France is"},
    ],
    stream=False,
    max_tokens=1024,
)
print(output.choices[0].message.content)
