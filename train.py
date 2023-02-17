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

with open("cat_to_name.json","r") as f:
    cat_to_name=json.load(f)

def get_input_args():
    parser=argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data_dir",type=str,default="./flowers/",help="Determines the directory")
    parser.add_argument("--save_dir",type=str,default="checkpoint.pth",help="To save the directory")
    parser.add_argument("--arch",type=str,default="vgg16",help="Determine which architectute to use")
    parser.add_argument("--learning_rate",type=float,default=0.0003,help="the rate at ehich model does its learning")
    parser.add_argument("--hidden_layer",type=int,default=4096,help="dictates hidden unit for hidden layer")
    parser.add_argument("--gpu",default=False,type=str,help="where to run model cpu or gpu")
    parser.add_argument("--epochs",type=int,default=3,help="number of cycle to train the model")
    parser.add_argument("--dropout",type=float,default=0.2,help="probalility rate for dropout")
    in_arg=parser.parse_args()
    gpu=in_arg.gpu
    print("Argument 1:",in_arg.data_dir)
    print("Argument 2:",in_arg.save_dir)
    print("Argument 3:",in_arg.arch)
    print("Argument 4:",in_arg.learning_rate)
    print("Argument 5:",in_arg.hidden_layer)
    print("Argument 6:",in_arg.gpu)
    print("Argument 7:",in_arg.epochs)
    print("Argument 8:",in_arg.dropout)
    return in_arg


data_dir="./flowers/"
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
#data_transforms 
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
                                     

# TODO: Load the datasets with ImageFolder
#image_datasets  
train_image_datasets=datasets.ImageFolder(train_dir,transform=train_transforms)
valid_image_datasets=datasets.ImageFolder(valid_dir,transform=valid_transforms)
test_image_datasets=datasets.ImageFolder(test_dir,transform=test_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders
trainloader=torch.utils.data.DataLoader(train_image_datasets,batch_size=64,shuffle=True)
testloader=torch.utils.data.DataLoader(test_image_datasets,batch_size=64,shuffle=True)
validloader=torch.utils.data.DataLoader(valid_image_datasets,batch_size=64,shuffle=True)

model=models.vgg16(pretrained=True)


classifier=nn.Sequential(OrderedDict([
                        ("fc1",nn.Linear(25088,4096)),
                        ("relu",nn.ReLU()),
                        ("dropout",nn.Dropout(0.2)),
                        ("fc2",nn.Linear(4096,102)),
                        ("output",nn.LogSoftmax(dim=1))]))
model.classifier=classifier
#criterion and optimizer
criterion=nn.NLLLoss()
optimizer=optim.Adam(model.classifier.parameters(),lr=3e-4)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# TODO: Do validation on the test set
# TODO: Do validation on the test set
epochs=3
steps=0
print_every=5
running_loss=0
for e in range(epochs):
    for images,labels in trainloader:
        steps+=1
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
            
test_accuracy=0
with torch.no_grad():
    for test_images,test_labels in testloader:
        model.eval()
        test_images=test_images.to(device)
        test_labels=test_labels.to(device)
   
        test_outputs=model.forward(test_images)
        test_loss=criterion(test_outputs,test_labels)
        ps=torch.exp(test_outputs)
        top_p,top_class=ps.topk(1,dim=1)
        equals=top_class==test_labels.view(*top_class.shape)
        test_accuracy+=torch.mean(equals.type(torch.FloatTensor))
    
    
    print("test accuracy:{:.2f} %".format(test_accuracy/len(testloader)*100))

    
# TODO: Save the checkpoint 
model.class_to_idx=train_image_datasets.class_to_idx
save_dir=""
checkpoint={"structure":"vgg16",
            "input_layer":25088,
            "classifier":model.classifier,
            "hidden_layer":4096,
            "dropout":0.2,
            "output_layer":102,
            "epochs":3,
            "gpu":model.to(device),
            "state_dict":model.state_dict(),
            "class_to_idx":model.class_to_idx,
            "optimizer_dict":optimizer.state_dict()}

torch.save(checkpoint,"checkpoint.pth")
            