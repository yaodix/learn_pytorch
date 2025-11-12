from transformers import AutoTokenizer

# 加载模型对应的tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
print(tokenizer)

# 计算文本的token数量
text = " The image depicts a close-up view of a vibrant pink flower, which appears to be a cosmos flower, \
surrounded by other flowers and plants in a garden setting. The cosmos flower is the central focus of the image, \
with its petals fully open and exhibiting a soft, delicate texture. The petals are a bright, almost neon pink color,\
  with a slight gradient towards the edges, giving the flower a radiant appearance. \
In the center of the cosmos flower, there is a small insect, likely a bee, which is actively collecting nectar from the flower. \
The bee is positioned on the flower's central part, surrounded by the flower's "
tokens = tokenizer.encode(text)

print(tokens)
print(f"Token数量: {len(tokens)}")