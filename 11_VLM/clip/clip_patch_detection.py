import math
import urllib.request
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

#@title Helper functions

#@markdown Some helper functions for loading, patchifying and visualizing images.

def load_image(img_path, resize=None, pil=False):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    if pil:
        return image
    return np.asarray(image).astype(np.float32) / 255.

def viz_patches(x, figsize=(20, 20), patch_idx=None, topk=None, t=5):
    # x: num_patches, 3, patch_size, patch_size
    n = x.shape[0]
    nrows = int(math.sqrt(n))
    _, axes = plt.subplots(nrows, nrows, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):            
        im = x[i].permute(1, 2, 0).numpy()
        im = (im * 255.).round().astype(np.uint8)
        if patch_idx is not None and i == patch_idx:
            im[0:t] = (255, 0, 0)
            im[im.shape[0]-t:] = (255, 0, 0)
            im[:, 0:t] = (255, 0, 0)
            im[:, im.shape[1]-t:] = (255, 0, 0)
        if topk is not None:
            if i in topk and i != patch_idx:
                im[0:t] = (255, 255, 0)
                im[im.shape[0]-t:] = (255, 255, 0)
                im[:, 0:t] = (255, 255, 0)
                im[:, im.shape[1]-t:] = (255, 255, 0)
        ax.imshow(im)
        ax.axis("off")
    # plt.show()
    plt.savefig("patches.png")

def patchify(image_path, resolution, patch_size, patch_stride=None):
    img_tensor = transforms.ToTensor()(load_image(image_path, resolution, True))
    if patch_stride is None:
        patch_stride = patch_size
    patches = img_tensor.unfold(
        1, patch_size, patch_stride).unfold(2, patch_size, patch_stride)
    patches = patches.reshape(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    return patches  # N, 3, patch_size, patch_size

#@title Image and Patch Settings { run: "auto" }
image_url = 'https://images2.minutemediacdn.com/image/upload/c_crop,h_706,w_1256,x_0,y_64/f_auto,q_auto,w_1100/v1554995050/shape/mentalfloss/516438-istock-637689912.jpg' #@param {type:"string"}
image_resolution =  900#@param {type:"integer"}
patch_size =  224#@param {type:"integer"}
# integer_input = 10 #@param {type:"integer"}
# integer_slider = 21 #@param {type:"slider", min:0, max:100, step:1}

# Download the image from the web.
image_path = 'image.png'
urllib.request.urlretrieve(image_url, image_path)

patches = patchify(image_path, image_resolution, patch_size)
print("patches: ", patches.shape)
viz_patches(patches, figsize=(8, 8))

#@title Detect

clip_model = "RN50" #@param ["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"]
image_caption = 'the dog' #@param {type:"string"}
topk =  6#@param {type:"integer"}

# Load CLIP model.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_model, device=device, jit=False)

text_input = clip.tokenize([image_caption]).to(device)

# Pad in case not equal to model expected input resolution.
p = model.visual.input_resolution - patch_size
patches_pad = torch.nn.functional.pad(
    patches, (p//2, p//2, p//2, p//2), "constant", 0).to(device)

with torch.no_grad():
    patch_embs = model.encode_image(patches_pad)
    text_embs = model.encode_text(text_input)
    patch_embs = patch_embs / patch_embs.norm(dim=-1, keepdim=True)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    sim = patch_embs @ text_embs.t()
    idx_max = sim.argmax().item()
    topk_idxs = torch.topk(sim.flatten(), topk)[-1].cpu().numpy().tolist()

viz_patches(patches, figsize=(10, 10), patch_idx=idx_max, topk=topk_idxs, t=int(0.05*patch_size))