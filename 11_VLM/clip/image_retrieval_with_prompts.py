#Import packages
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from datasets import load_dataset
import faiss
import numpy as np
from tqdm import tqdm

#Define device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#Load CLIP model, processor and tokenizer
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

#Load cifar10 dataset
dataset = load_dataset("cifar10")

#Limit size to 10000 images
from datasets import Dataset
train_dataset = dataset["train"].filter(lambda example, idx: idx < 10000, with_indices=True)
print(train_dataset)

#Display an image
# display(train_dataset[0]['img'])
#Add a vector to FAISS index
def add_vector_to_index(embedding, index):
    #convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    vector = np.float32(vector)
    #Normalize vector: important
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

#Extract features of a given image
def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        return image_features

#FAISS index
index = faiss.IndexFlatL2(512)

#Process the dataset to extract all features and store in index
for image in tqdm(train_dataset):
    clip_features = extract_features_clip(image['img'])
    add_vector_to_index(clip_features,index)
    
#Write index locally. Not needed after but can be useful for future retrieval
faiss.write_index(index,"clip.index")

#Let's search images of dogs
prompt="a photo of a dog"

#Tokenize the prompt and extract features
text_token = tokenizer([prompt], return_tensors="pt").to(device)
text_features = model.get_text_features(**text_token)

#Preprocess the vector before search in FAISS index
text_np = text_features.detach().cpu().numpy()
text_np = np.float32(text_np)
faiss.normalize_L2(text_np)

#Search the top 5 images
distances, indices = index.search(text_np, 5)
print('distances',distances)
print('indices' ,indices)

#For each top-5 results, compute similarity score between 0 and 1, print indice, similarity score and display image 
for i,v in enumerate(indices[0]):
    sim = (1/(1+distances[0][i])*100)
    print(f"Indice: {v} , Similarity score: {sim}")
    img_resized = train_dataset[int(v)]['img'].resize((200, 200))
    # display(img_resized)
    img_resized.save(f"dog_{i}.jpg")
