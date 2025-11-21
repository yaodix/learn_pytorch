#Import packages
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

#Define device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#Load CLIP model, processor and tokenizer
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

#Load cifar10 dataset
dataset = load_dataset("cifar10")

#Display the list of labels
labels = dataset["train"].features["label"].names
print(f"labels: {labels}")

new_labels = [f"a photo of {label}" for label in labels] # upgrade acc 1.6%
print(f"new_labels: {new_labels}")

#Take the first image in the training set
image = dataset['train'][0]['img']

#Function to classify an image among the list of labels
def classify(image, labels):
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1) 
    label = probs.argmax()
    return label
# demo
l = classify(image,labels)
#Display the first image in the dataset and its predicted class (i.e. airplane, which is correct)
# display(image)
print(labels[l])

predictions= [] 
ground_truth= [d['label'] for d in dataset['test']]

for img in tqdm(dataset['test']):
    pred = classify(img['img'], labels)
    predictions.append(pred.item())
    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
y_true = ground_truth 
y_pred = predictions 

# Compute accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')

precision = precision_score(y_true, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')

recall = recall_score(y_true, y_pred, average='weighted')
print(f'Recall: {recall:.4f}')

f1 = f1_score(y_true, y_pred, average='weighted')
print(f'F1 Score: {f1:.4f}')

# Generate classification report
print('\nClassification Report:')
print(classification_report(y_true, y_pred))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print('\nConfusion Matrix:')
print(conf_matrix)