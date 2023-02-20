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
from PIL import Image
import argparse



with open("cat_to_name.json","r") as f:
    cat_to_name=json.load(f)


parser=argparse.ArgumentParser(description="train.py")
parser.add_argument("data_dir",type=str,default="./flowers/",help="Determines the directory")
parser.add_argument("--save_dir",type=str,default="checkpoint.pth",help="To save the directory")
parser.add_argument("--arch",type=str,default="vgg16",help="Determine which architectute to use")
parser.add_argument("--learning_rate",type=float,default=0.0003,help="the rate at ehich model does its learning")
parser.add_argument("--hidden_layer",type=int,default=4096,help="dictates hidden unit for hidden layer")
parser.add_argument("--gpu",default=False,type=str,help="where to run model cpu or gpu")
parser.add_argument("--epochs",type=int,default=3,help="number of cycle to train the model")
parser.add_argument("--dropout",type=float,default=0.2,help="probalility rate for dropout")
in_arg=parser.parse_args()
in_arg=parser.parse_args()
path=in_arg.data_dir
new_path=in_arg.save_dir
lr=in_arg.learning_rate
structure=in_arg.arch
dropout=in_arg.dropout
device=in_arg.gpu
epochs=in_arg.epochs    
hidden_layer=in_arg.hidden_layer

arch={"vgg16":25088,
      "densenet121":1024,
      "resnet50":2048,
      "alexnet":9216}

def data_loader(path="./flowers"):
    data_dir=path
    train_dir=data_dir+"/train"
    valid_dir=data_dir+"/valid"
    test_dir=data_dir+"/test"
    train_transforms=transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],
                                                              [0.229,0.224,0.225])])
    valid_transforms=transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],
                                                             [0.229,0.224,0.225])])
    test_transforms=transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],
                                                             [0.229,0.224,0.225])])
                                     

 
    train_image_datasets=datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_image_datasets=datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_image_datasets=datasets.ImageFolder(test_dir,transform=test_transforms)
    
   
    trainloader=torch.utils.data.DataLoader(train_image_datasets,batch_size=64,shuffle=True)
    testloader=torch.utils.data.DataLoader(test_image_datasets,batch_size=64,shuffle=True)
    validloader=torch.utils.data.DataLoader(valid_image_datasets,batch_size=64,shuffle=True)
  
    return trainloader,validloader,testloader,train_image_datasets

trainloader,validloader,testloader,train_image_datasets=data_loader(path="./flowers")
     

def network(structure="vgg16",dropout=0.2,lr=0.0003,device="gpu"):
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
                        ("fc1",nn.Linear(arch[structure],4096)),
                        ("relu",nn.ReLU()),
                        ("dropout",nn.Dropout(0.2)),
                        ("fc2",nn.Linear(4096,102)),
                        ("output",nn.LogSoftmax(dim=1))]))
        model.classifier=classifier
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return model
print(network())
model=network(structure,dropout=0.2,lr=0.0003,device="gpu")
criterion=nn.NLLLoss()    
optimizer=optim.Adam(model.classifier.parameters(),lr=3e-4)
              
def train_network(model=model, trainloader=trainloader, criterion=criterion, optimizer=optimizer, epochs=epochs, device="gpu"):
    optimizer=optim.Adam(model.classifier.parameters(),lr=3e-4)
    steps=0
    print_every=5
    running_loss=0
    for e in range(epochs):
        for images,labels in trainloader:
            steps+=1
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs=model.forward(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        #calculate valid loss and accuracy
            if steps%print_every==0:
                model.eval()
                valid_loss=0
                val_accuracy=0
                for val_images,val_labels in validloader:
                    val_images=val_images.to(device)
                    val_labels=val_labels.to(device)
                    with torch.no_grad():
                        valid_outputs=model.forward(val_images)
                        valid_loss=criterion(valid_outputs,val_labels)
                        ps=torch.exp(valid_outputs)
                        top_p,top_class=ps.topk(1,dim=1)
                        equals=top_class==val_labels.view(*top_class.shape)
                        val_accuracy+=torch.mean(equals.type(torch.FloatTensor))
            #print loss and accuracy
                print("Epoch: {}/{}..".format(e+1,epochs),
                    "Train loss: {:.3f}..".format(running_loss/print_every),
                    "Validation loss: {:.3f}..".format(valid_loss/len(validloader)),
                    "Validation Accuracy: {:.3f}..".format(val_accuracy/len(validloader)))
                running_loss=0
                model.train()

train_network(model, trainloader, criterion, optimizer, epochs, device)
def save_checkpoint(model,new_path="checkpoint.pth",structure="vgg16",hidden_layer=4096,lr=0.0003,epochs=3):
    model.class_to_idx=train_image_datasets.class_to_idx
    torch.save({"structure":structure,
               
                "hidden_layer":hidden_layer,
                "dropout":0.2,
                "lr":lr,
                "epochs":3,
                "gpu":model.to(device),
                "state_dict":model.state_dict(),
                "class_to_idx":model.class_to_idx},
                new_path)
print("model is trained")
    
