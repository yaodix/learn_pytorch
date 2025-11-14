from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
print(model)

inputs = processor(images=image, return_tensors="pt")  # dict{pixel_values: shape[1, 3, 224, 224]}
outputs = model(**inputs)  # 等价于outputs = model(pixel_values=inputs['pixel_values'])
logits = outputs.logits   # [1, 1000]
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
