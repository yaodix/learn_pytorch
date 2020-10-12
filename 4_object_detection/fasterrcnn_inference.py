import torch
import torchvision
import argparse
import cv2
import numpy as np
import sys
# sys.path.append('./')
import coco_names
import random


def get_args(argsInput ):
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')

    parser.add_argument('image_path', type=str, help='image path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='model')
    parser.add_argument('--score', type=float, default=0.8, help='objectness score threshold')

    if argsInput == '':
        args = parser.parse_args()
    else:
        args = parser.parse_args(argsInput)
    return args


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)


def detect(argsInput):
    args = get_args(argsInput)
    input = []
    num_classes = 91
    names = coco_names.names

    # Model creating
    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=True) #__dict__属性
    # print('detecion.__detect__ start')
    # print(torchvision.models.detection.__dict__)
    # print('detecion.__detect__ end')
    model = model.cuda()
    model.eval()

    src_img = cv2.imread(args.image_path)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()  #permute
    input.append(img_tensor)
    out = model(input)
    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']

    for idx in range(boxes.shape[0]):
        if scores[idx] >= args.score:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
            cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

    cv2.imshow('result', src_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = {
        'C:\\MyData\\VOCdevkit\\VOC2007\\JPEGImages\\000222.jpg'
    }
    detect(args)
