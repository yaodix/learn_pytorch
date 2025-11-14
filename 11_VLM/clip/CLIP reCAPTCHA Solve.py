#@title Install dependencies

#@markdown Please execute this cell by pressing the _Play_ button 
#@markdown on the left.

#@markdown **Note**: This installs the software on the Colab 
#@markdown notebook in the cloud and not on your computer.

%%capture
!pip install ftfy regex tqdm matplotlib bs4
!pip install git+https://github.com/openai/CLIP.git

import urllib.request
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

#@title Helper functions

#@markdown Some helper functions for loading, patchifying and visualizing images.

def load_image(img_path, resize=None, pil=False):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    if pil:
        return image
    image = np.asarray(image).astype(np.float32) / 255.
    return image

def viz_patches(x, figsize=None, topk=None, t=5, title=None):
    color = (0, 255, 0)
    n = x.shape[0]
    images = []
    for i in range(n):
        im = x[i].permute(1, 2, 0).numpy()
        im = (im * 255.).round().astype(np.uint8)
        if topk is not None:
            if i in topk:
                im[0:t] = color
                im[im.shape[0]-t:] = color
                im[:, 0:t] = color
                im[:, im.shape[1]-t:] = color
        images.append(torch.from_numpy(im).permute(2, 0, 1))
    im = make_grid(images, 3).permute(1, 2, 0).numpy()
    plt.figure(figsize=figsize)
    plt.imshow(im)
    plt.axis('off')
    if title is not None:
        plt.title(title, fontsize=20)
    plt.show()

def patchify(image_path, resolution, patch_size, patch_stride=None, resize=None):
    img_tensor = torch.from_numpy(load_image(image_path, resolution)).permute(2, 0, 1)
    if patch_stride is None:
        patch_stride = patch_size
    patches = img_tensor.unfold(
        1, patch_size, patch_stride).unfold(2, patch_size, patch_stride)
    patches = patches.reshape(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    if resize is not None:
        patches = F.interpolate(
            patches,
            (resize, resize),
            mode='bilinear',
            align_corners=False)
    return patches  # N, 3, patch_size, patch_size
  
  #@title Query reCAPTCHA API

#@markdown You might have to click the *Play* button more than once if the URL
#@markdown returns a 404.

URL = "https://www.google.com/recaptcha/api/fallback?k=6LewPtQSAAAAAIvk6kmw1mVSYVUvd2Ev5MpenlHk"
url_contents = urllib.request.urlopen(URL).read()
soup = BeautifulSoup(url_contents, "html")
instruction = soup.find("div", {"class": "rc-imageselect-desc-no-canonical"}).get_text()
image = soup.find("img", {"class": "fbc-imageselect-payload"})
image_url = f"https://www.google.com/{image['src']}"
image_path = 'image.png'
urllib.request.urlretrieve(image_url, image_path)

image_np = load_image(image_path)
print(f"Captcha Image Resolution: {image_np.shape}")
plt.figure(figsize=(10, 10))
plt.imshow(image_np)
plt.title(instruction, fontsize=20)
plt.axis("off")
plt.show()