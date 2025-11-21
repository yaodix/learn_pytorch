#Import packages
import torch
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor
from datasets import load_dataset
import faiss
import numpy as np
import time

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


if __name__ == "__main__":
    #Load index
  #Define device
  device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

  #Load CLIP model and processor
  processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
  model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

  #Load cifar10 dataset
  dataset = load_dataset("cifar10")

  # Display an image
  # img = (dataset['train'][0]['img'])
  # img.save("cifar.jpg")

  # Create FAISS index
  index = faiss.IndexFlatL2(512)

  #Process the dataset to extract all features and store in index
  for image in dataset['test']:
      clip_features = extract_features_clip(image['img'])
      add_vector_to_index(clip_features, index)
      
  #Write index locally. Not needed after but can be useful for future retrieval
  faiss.write_index(index,"clip.index")
  # Retrieve the vectors
  
  vectors = index.reconstruct_n(0, 10000)

  #Define clusters and parameters
  x= vectors
  ncentroids = 10
  niter = 50
  verbose = True
  d = x.shape[1]

  t0=time.time()
  #Launch clustering
  kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
  kmeans.train(x)
  print('Clustering done in ', time.time()-t0)

  #cluster assignments: [array of vectors, array of indices (cluster assigned to a given vector)]
  cluster_assignments = kmeans.assign(vectors)
  print(f"cluster_assignments: {cluster_assignments}")
  #Print indices only
  clusters_ind = cluster_assignments[1]
  print(f"clusters_ind: {clusters_ind}")

  distribution = np.zeros((10, 10), dtype=int)
  print(distribution)
  
  for vector_index, cluster_nb in enumerate(clusters_ind):
    label = dataset['test'][vector_index]['label']
    distribution[cluster_nb][label]+=1
 
  print(distribution)
  
  row_sums = np.sum(distribution, axis=1)
  print("Number of images per cluster:")
  print(row_sums)
  
  