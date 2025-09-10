from transformers import pipeline
import torch  # 新增导入torch

# 打印CUDA信息
if torch.cuda.is_available():
    print(f"可用CUDA设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"设备ID {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA不可用")

classifier = pipeline("sentiment-analysis")
print(classifier("I love this movie!"))