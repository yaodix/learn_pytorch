# Analyzing Artistic Styles with Multimodal Embeddings

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import fiftyone as fo # base library and app
import fiftyone.zoo as foz # zoo datasets and models
import fiftyone.brain as fob # ML routines
from fiftyone import ViewField as F # for defining custom views
import fiftyone.utils.huggingface as fouh # for loading datasets from Hugging Face

dataset = fouh.load_from_hub(
    "Koshti10/Kitti-Images", ## repo_id
    format="parquet", ## for Parquet format
    # classification_fields=["artist", "style", "genre"], # columns to store as classification fields
    max_samples=500, # number of samples to load
    name="Kitti-Images", # name of the dataset in FiftyOne
)

# print(dataset)


# artists = dataset.distinct("artist.label")
# print(artists)

fob.compute_similarity(
    dataset, 
    model="zero-shot-classification-transformer-torch", ## type of model to load from model zoo
    name_or_path="openai/clip-vit-base-patch32", ## repo_id of checkpoint
    embeddings="clip_embeddings", ## name of the field to store embeddings
    brain_key="clip_sim", ## key to store similarity index info
    batch_size=32, ## batch size for inference
    )
import time 

print("end")
session = fo.launch_app(dataset)

print("应用已启动，按Ctrl+C退出...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n退出应用")