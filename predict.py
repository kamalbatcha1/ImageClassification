import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
from torch import optim
from torch import nn
from torchvision import datasets,transforms,models
from collections import OrderedDict
import torch.nn.functional as F
import json
import PIL
import argparse

parser=argparse.ArgumentParser()

parser.add_argument("--json_file",type=str,default="cat_to_name.json",help="Allows user to enter custom JSON file ")
parser.add_argument("image_path",type=str,default="flowers/test/13/image_05775.jpg",help="test image")
parser.add_argument("--checkpoint",type=str,default="checkpoint.pth",help="load/built checkpoint")
parser.add_argument("--topk",type=int,default=5,help="top K prediction")
parser.add_argument("--gpu",type=str,default=False,help="run model cpu or gpu")
 
in_arg=parser.parse_args()

json_file=in_arg.json_file

image_path=in_arg.image_path
new_path=in_arg.checkpoint
topK=in_arg.topk
gpu=in_arg.gpu
 

with open(json_file,"r") as f:
    cat_to_name=json.load(f)

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(new_path="save_dir"):
    checkpoint=torch.load("save_dir")
    structure=checkpoint["structure"]
    hidden_layer=checkpoint["hidden_layer"]
    epochs=checkpoint["epochs"]
    gpu=checkpoint["gpu"]
    
    dropout=checkpoint["dropout"]
    state_dict=checkpoint["state_dict"]
    class_to_idx=checkpoint["class_to_idx"]
    arch={"vgg16":25088,
        "densenet121":1024,
        "resnet50":2048,
        "alexnet":9216}
    if structure=="vgg16":
        model=models.vgg16(pretrained=True)
        hidden_layer=4096
    elif structure=="densenet121":
        model=models.densenet121(pretrained=True)
        hidden_layer=120
    elif structure=="resnet50":
        model=models.resnet50(pretrained=True)
        hidden_layer=240
    elif structure=="alexnet":
        model=models.alexnet(pretrained=True)
        hiddenlayer=1024
    else:
        print("choose a model {} between vgg16,densenet121,alexnet,resnet therefore vgg16 is a pretty good        model".format(structure))
    for param in model.parameters():
        param.requires_grad=False
        
        classifier=nn.Sequential(OrderedDict([
                        ("fc1",nn.Linear(arch[structure],hidden_layer)),
                        ("relu",nn.ReLU()),
                        ("dropout",nn.Dropout(dropout)),
                        ("fc2",nn.Linear(hidden_layer,102)),
                        ("output",nn.LogSoftmax(dim=1))]))
        model.classifier=classifier
    model.load_state_dict(state_dict)
    return model
loaded_checkpoint=load_checkpoint("save_dir")
print(loaded_checkpoint)


from PIL import Image
def process_image(image):
    
    # TODO: Process a PIL image for use in a PyTorch model
    image=Image.open(image)
    transform=transforms.Compose([transforms.RandomRotation(30),
                                  transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485,0.456,0.406],
                                                       [0.229,0.224,0.225])])
    trans_image=transform(image)
    return trans_image
def gpu(device):
    if not device:
        return torch.device("cpu")
    device=torch.device("cuda:0" if torch.cuda is available() else"cpu")
    if device=="cpu":
        print("CUDA was not available using cpu")
    return device
gpu(device=in_arg.gpu)
image_path="flowers/test/13/image_05775.jpg"

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image=process_image(image_path)
    image=image.float().unsqueeze_(0)
    model.to("cuda")
    model.eval()
    with torch.no_grad():
        output=model.forward(image.cuda())
    prediction=torch.exp(output).data
    return prediction.topk(topk)  
print("the prediction is")
model=loaded_checkpoint
probs, classes = predict(image_path, model)
print(probs)
print(classes)

