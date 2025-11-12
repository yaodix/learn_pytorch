import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from transformers.image_utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Quantization for efficiency
quant_config = BitsAndBytesConfig(load_in_4bit=True)
model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(model_name, quantization_config=quant_config).to(device)
processor = AutoProcessor.from_pretrained(model_name)

# Load image
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = load_image(image_url)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe the image?"}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
print(f"prompt: {prompt}")
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(device)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]

# Extract only the assistant response
assistant_response = generated_texts.split("Assistant:")[-1].strip()

print(assistant_response)