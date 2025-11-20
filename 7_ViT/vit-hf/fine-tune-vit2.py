# https://github1s.com/huggingface/computer-vision-course/tree/main

import io
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from huggingface_hub import notebook_login

from datasets import load_dataset, DatasetDict

from transformers import AutoImageProcessor, ViTForImageClassification

from transformers import Trainer, TrainingArguments

import evaluate

dataset = load_dataset('enterprise-explorers/oxford-pets')

def show_samples(ds,rows,cols):
    samples = ds.shuffle().select(np.arange(rows*cols)) # selecting random images
    fig = plt.figure(figsize=(cols*4,rows*4))
    # plotting
    for i in range(rows*cols):
        img = samples[i]['image']
        label = samples[i]['label']
        fig.add_subplot(rows,cols,i+1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
    # plt.savefig('samples.png')
def show_predictions(rows,cols):
    samples = our_dataset['test'].shuffle().select(np.arange(rows*cols))
    processed_samples = samples.with_transform(transforms)
    predictions = trainer.predict(processed_samples).predictions.argmax(axis=1) # predicted labels from logits
    fig = plt.figure(figsize=(cols*4,rows*4))
    for i in range(rows*cols):
        img_bytes = samples[i]['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        prediction = predictions[i]
        label = f"label: {samples[i]['label']}\npredicted: {id2label[prediction]}"
        fig.add_subplot(rows,cols,i+1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
if __name__ == '__main__':
  show_samples(dataset['train'],rows=3,cols=5)

  split_dataset = dataset['train'].train_test_split(test_size=0.2) # 80% train, 20% evaluation
  eval_dataset = split_dataset['test'].train_test_split(test_size=0.5) # 50% validation, 50% test

  # recombining the splits using a DatasetDict
  our_dataset = DatasetDict({
      'train': split_dataset['train'],
      'validation': eval_dataset['train'],
      'test': eval_dataset['test']
  })
  model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels = len(labels),
    id2label = id2label,
    label2id = label2id,
    ignore_mismatched_sizes = True
)

    for name,p in model.named_parameters():
        if not name.startswith('classifier'):
            p.requires_grad = False
    print(our_dataset)