import numpy as np
import torch,torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
from torch import optim
import cv2
import os
from PIL import Image
import pandas as pd
import seaborn as sns
import torch.nn.functional as F



#pre processing
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
     transforms.RandomAffine(degrees=(25)),
     transforms.RandomRotation(25),
     transforms.RandomHorizontalFlip(0.5),
     transforms.RandomVerticalFlip(0.5),
     transforms.Normalize((0.5,0.5,0.5),(1,1,1))
])


batch_size=64
epochs=100
num_classes=16




# ## model creation

# In[12]:


model=torchvision.models.resnet18(pretrained=True)
num_ftrs=model.fc.in_features
model.fc=nn.Linear(num_ftrs,16)

model=model.to(device)
model


# In[13]:


summary(model,input_size=(3,224,224))


# In[14]:


loss_fn=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)


# In[15]:


path="C:\\Users\\NOUFAL\\OneDrive\\Documents\\PROJECT\\resnet_18_epochs"


# In[16]:


model_path="C:\\Users\\NOUFAL\\OneDrive\\Documents\\PROJECT\\resnet_18_epochs\\epoch-61.pt"
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))


# #### training

# In[ ]:


    


# In[17]:


model.eval()


# In[19]:






def pred(img_path,transform):
    image=Image.open(img_path)
    image_tensor=transform(image).float()
    image_tensor=image_tensor.unsqueeze(0)
    image_tensor=image_tensor.to(device)
    output=model(image_tensor)
    index=torch.argmax(output)
    pred=test_data.classes[index]
    plt.imshow(cv2.imread(img_path))
    probs, _ = torch.max(F.softmax(output,dim=1),1)
    print(probs)
    if probs <0.93:
        print("not defined")
    else:    
        return pred
test="C:\\Users\\NOUFAL\\Downloads\\th (2).jpg"
print("The predicted class is :")
pred(test,transform)




# In[ ]:




