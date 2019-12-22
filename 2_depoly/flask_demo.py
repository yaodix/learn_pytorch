from flask import Flask
from flask import jsonify
from flask import Response
from flask import request
import json
import io
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


app = Flask(__name__)
imagenet_cls_index  =json.load(open("C:\\MyData\\imagenet_class_index.json"))
model = models.densenet121(pretrained=True)   #在预测函数外部load 模型
model.eval()

def transform_image(image_bytes):
    my_trans = transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485,0.456,0.460],[0.229,0.224,0.225])

                                   ])
    image = Image.open(io.BytesIO(image_bytes))
    return  my_trans(image).unsqueeze(0)


def get_prediction(image_butes):
    tensor = transform_image(image_bytes=image_butes)
    outputs = model.forward(tensor)
    _,y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return  imagenet_cls_index[predicted_idx]

@app.route('/predict',methods = ['POST'])   #post --form_data(set file) / key:file ,value:path
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        cls_id,cls_name = get_prediction(image_butes=img_bytes)

        return jsonify({'class_id':cls_id,'class_name':cls_name})



if __name__ == '__main__':
    app.run()