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


checkpoint="checkpoint.pth"
def get_input_args():
    parser=argparse.ArgumentParser()

    parser.add_argument("--json_file",type=str,default="cat_to_name.json",help="Allows user to enter custom JSON file ")
    parser.add_argument("--image_path",type=str,default="flowers/test/13/image_05775.jpg",help="test image")
    parser.add_argument("--checkpoint",type=str,default="checkpoint.pth",help="load/built checkpoint")
    parser.add_argument("--topk",type=int,default=5,help="top K prediction")
    parser.add_argument("--gpu",type=str,default=False,help="run model cpu or gpu")
 
    in_arg=parser.parse_args()
    print("Argument 1:",in_arg.json_file)
    print("Argument 2:",in_arg.test_file)
    print("Argument 3:",in_arg.checkpoint)
    print("Argument 4:",in_arg.topk)
    print("Argument 5:",in_arg.gpu)
    return in_arg

with open("cat_to_name.json","r") as f:
    cat_to_name=json.load(f)

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    checkpoint=torch.load("checkpoint.pth")
    structure=checkpoint["structure"]
    input_layer=checkpoint["input_layer"]
    hidden_layer=checkpoint["hidden_layer"]
    output_layer=checkpoint["output_layer"]
    epochs=checkpoint["epochs"]
    gpu=checkpoint["gpu"]
    
    dropout=checkpoint["dropout"]
    optimizer=checkpoint["optimizer_dict"]
    state_dict=checkpoint["state_dict"]
    class_to_idx=checkpoint["class_to_idx"]
    model=models.vgg16(pretrained=True)
    classifier=nn.Sequential(OrderedDict([
                        ("fc1",nn.Linear(25088,4096)),
                        ("relu",nn.ReLU()),
                        ("dropout",nn.Dropout(0.2)),
                        ("fc2",nn.Linear(4096,102)),
                        ("output",nn.LogSoftmax(dim=1))]))
    model.classifier=classifier
    model.load_state_dict(state_dict)
    return model
loaded_checkpoint=load_checkpoint("checkpoint.pth")
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
image_path="flowers/test/13/image_05775.jpg"
topk=5
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image=process_image(image_path)
    image=image.float().unsqueeze_(0)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output=model.forward(image.cuda())
    prediction=torch.exp(output).data
    return prediction.topk(topk)
    
image_path="flowers/test/13/image_05775.jpg"
model=loaded_checkpoint
probs, classes = predict(image_path, model)
print(probs)
print(classes)

