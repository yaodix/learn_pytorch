#Import packages
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import faiss
import numpy as np
import requests
from datasets import load_dataset, Dataset
from tqdm import tqdm

#Define device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#Load CLIP model, processor and tokenizer
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

#Load tiny-imagenet dataset
dataset = load_dataset("zh-plus/tiny-imagenet")

#Display an image
# display(dataset['valid'][0]['image'])

#Keep only validation set
valid_dataset = dataset["valid"]
print(valid_dataset)
#Add a vector to FAISS index
def add_vector_to_index(embedding, index):
    #convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    vector = np.float32(vector)
    #Normalize vector
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

#Extract features of a given image
def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        return image_features

#Create FAISS index
index = faiss.IndexFlatL2(512)

#Process the dataset to extract the features and store them into the FAISS index
for image in tqdm(valid_dataset):
    clip_features = extract_features_clip(image['image'])
    add_vector_to_index(clip_features,index)
    
#Write index locally. Not needed after but can be useful for future retrieval
faiss.write_index(index,"clip.index")

#Fetch an image of two cats that are not in the dataset
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
input_image = Image.open(requests.get(url, stream=True).raw)
# display(input_image)

#Extract features of the input image
input_features = extract_features_clip(input_image)

#Preprocess the vector before searching the FAISS index
input_features_np = input_features.detach().cpu().numpy()
input_features_np = np.float32(input_features_np)
faiss.normalize_L2(input_features_np)

#Search the top 5 images
distances, indices = index.search(input_features_np, 5)
print('distances',distances)
print('indices' ,indices)


#For each top-5 results, compute similarity score between 0 and 1, print indice, similarity score and display image 
for i,v in enumerate(indices[0]):
    sim = (1/(1+distances[0][i])*100)
    print(f"Indice: {v} , Similarity score: {sim}")
    img_resized = valid_dataset[int(v)]['image'].resize((200, 200))
    # display(img_resized)