#通过实现Grad-CAM学习module中的forward_hook和backward_hook函数

import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class GradCam():
    def __init__(self,cam_size = 224,cls_num=2,out_dir="/"):
        super(GradCam, self).__init__()
        self.fmap_block = list()
        self.grad_block = list()
        self.cls_num = cls_num
        self.cam_size = cam_size
    def backward_hook(self,module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())


    def farward_hook(self,module, input, output):
        self.fmap_block.append(output)


    def show_cam_on_image(self,img, mask, out_dir):
        heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)

        path_cam_img = os.path.join(out_dir, "cam.jpg")
        path_raw_img = os.path.join(out_dir, "raw.jpg")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(path_cam_img, np.uint8(255 * cam))
        cv2.imwrite(path_raw_img, np.uint8(255 * img))


    def comp_class_vec(self,output_vec, index=None,device = 'cuda:0'):
        """
        计算类向量
        :param ouput_vec: tensor
        :param index: int，指定类别
        :return: tensor
        """
        if not index:
            index = np.argmax(output_vec.cpu().data.numpy())
        else:
            index = np.array(index)
        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros(1, self.cls_num).scatter_(1, index, 1)
        one_hot.requires_grad = True
        one_hot = one_hot.cuda()
        class_vec = torch.sum(one_hot * output_vec)  # one_hot = 11.8605

        return class_vec


    def gen_cam(self,feature_map, grads):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

        weights = np.mean(grads, axis=(1, 2))  #

        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (self.cam_size , self.cam_size ))
        cam -= np.min(cam)
        cam /= np.max(cam)

        return cam


if __name__ == '__main__':

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_img = os.path.join(BASE_DIR, "..", "..", "Data", "cam_img", "test_img_8.png")
    path_net = os.path.join(BASE_DIR, "..", "..", "Data", "net_params_72p.pkl")
    output_dir = os.path.join(BASE_DIR, "..", "..", "Result", "backward_hook_cam")

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img)
    net = Net()
    net.load_state_dict(torch.load(path_net))

    # 注册hook
    net.conv2.register_forward_hook(farward_hook)
    net.conv2.register_backward_hook(backward_hook)

    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (32, 32))) / 255
    show_cam_on_image(img_show, cam, output_dir)