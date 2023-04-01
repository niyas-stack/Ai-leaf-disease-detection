import numpy as np
import matplotlib.pyplot as plt
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


for epoch in range(61,epochs):
    for i,(images,labels) in enumerate(trainloader):
        print("steps in epoch : ",i+1)
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        loss=loss_fn(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    print('Epoch [{}/{}],loss{}'.format(epoch+1,epochs,loss.item()))
    torch.save(model.state_dict(),os.path.join(path,'epoch-{}.pt'.format(epoch+1)))
    


# In[17]:


model.eval()


# In[19]:


with torch.no_grad():
    correct=0
    total=0
    for i,(images,labels) in enumerate(valloader):
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        pred=torch.max(outputs,1)[1]
        total+=labels.size(0)
        correct +=(pred==labels).sum().item()
        accu=correct/total
    print("accuracy :%.2f"%accu)


# In[20]:


from sklearn.metrics import f1_score

# Get predictions and true labels for all images in the validation set
preds = []
labels_all = []
for i, (images, labels) in enumerate(valloader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    preds.append(torch.max(outputs, 1)[1])
    labels_all.append(labels)

# Concatenate all predictions and labels into a single tensor
preds_all = torch.cat(preds, dim=0)
labels_all = torch.cat(labels_all, dim=0)

# Calculate the F1 score
f1 = f1_score(labels_all.cpu().numpy(), preds_all.cpu().numpy(), average='macro')

print("Accuracy: {:.2f}%".format(accu*100))
print("F1 Score: {:.2f}".format(f1))


# #### Prediction
# 

# In[102]:


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


# In[67]:


model_path="C:\\Users\\user\\Documents\\project\\epochs\\epoch-9.pt"
model.load_state_dict(torch.load(model_path))


# ### confusion matrix

# In[21]:


confusion_matrix = np.zeros((num_classes, num_classes)) #used to create a new array of given shapes and types filled with zero values

with torch.no_grad():

   for i, (images,labels) in enumerate(testloader):

       images = images.to(device)

       labels = labels.to(device)

       outputs = model(images)

       _, preds = torch.max(outputs, 1)

       for t, p in zip(labels.view(-1), preds.view(-1)):

               confusion_matrix[t.long(), p.long()] += 1



plt.figure(figsize=(5,5))



class_names = list(classes.values())

df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)

heatmap = sns.heatmap(df_cm, annot=True, fmt="d") # annot only help to add numeric value on python heatmap cell but fmt parameter allows to add string (text) values on the cell



heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)

plt.title('confusion matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label')


# In[ ]:


num_classes = 2

confusion_matrix = np.zeros((num_classes, num_classes)) #used to create a new array of given shapes and types filled with zero values

with torch.no_grad():

   for i, (images,labels) in enumerate(testloader):

       images = images.to(device)

       labels = labels.to(device)

       outputs = model(images)

       _, preds = torch.max(outputs, 1)

       for t, p in zip(labels.view(-1), preds.view(-1)):

               confusion_matrix[t.long(), p.long()] += 1



plt.figure(figsize=(8,8))



class_names = list(classes.values())

df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)

heatmap = sns.heatmap(df_cm, annot=True, fmt="d") # annot only help to add numeric value on python heatmap cell but fmt parameter allows to add string (text) values on the cell



heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)

plt.title('confusion matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label')




# In[35]:


import os

path = "C:\\Users\\NOUFAL\\OneDrive\\Documents\\PROJECT\\data_split\\train\\bean_angular_leaf_spot" # replace with the path to your folder
new_prefix = "image" # replace with the new prefix you want to add to the filenames
counter = 1

for filename in os.listdir(path):
    old_path = os.path.join(path, filename)
    new_filename = new_prefix + str(counter) + ".jpg" # replace ".jpg" with the appropriate file extension
    new_path = os.path.join(path, new_filename)
    os.rename(old_path, new_path)
    counter += 1


# In[29]:


n=12
for i in range (5,n):
    print('i [{}/{}]'.format(i+1,n))
    


# In[ ]:




