from transformers import pipeline
import torch  # 新增导入torch

# Initialize the pipeline with a VLM
pipe = pipeline("image-text-to-text", "HuggingFaceTB/SmolVLM2-2.2B-Instruct", device_map="auto", dtype=torch.bfloat16)

# Define your conversation with an image
messages = [
     {
         "role": "user",
         "content": [
             {
                 "type": "image",
                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
             },
             {"type": "text", "text": "Describe this image."},
         ],
     }
 ]

outputs = pipe(text=messages, max_new_tokens=60, return_full_text=False)

# Generate response - pipeline handles multimodal inputs automatically
response = pipe(messages, max_new_tokens=128, temperature=0.7)

print(response[0]['generated_text'][-1]['content'])  # Print the model's description