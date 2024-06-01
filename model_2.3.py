import os
import nibabel as nib
from scipy import ndimage
import numpy as np
from numpy import random
import torch
import matplotlib.pyplot as plt
import torchio as tio
import cv2

data_path = r"C:\Users\punnut\Downloads\final_dataset1_preprocessed2.1_64"
AD_class = []
CN_class = []
MCI_class = []
MCI_AD_class = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".nii"):
            pathList = root.split(os.sep)
            for i in pathList:
                if i == "AD":
                    AD_class.append(os.path.join(root,file))
                    #print(os.path.join(root,file))
                elif i == "CN":
                    CN_class.append(os.path.join(root, file))
                    #print(os.path.join(root, file))
                elif i == "MCI":
                    MCI_class.append(os.path.join(root, file))
                    #print(os.path.join(root, file))
                elif i == "MCI_AD":
                    MCI_AD_class.append(os.path.join(root, file))
                    #print(os.path.join(root, file))

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    print(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 64
    desired_height = 64
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    #volume = normalize(volume)
    # Resize width, height and depth
    #volume = resize_volume(volume)
    return volume

num_classes = 3
#AD_scans = np.array([process_scan(path) for path in AD_class])
CN_scans = np.array([process_scan(path) for path in CN_class])
MCI_scans = np.array([process_scan(path) for path in MCI_class])
MCI_AD_scans = np.array([process_scan(path) for path in MCI_AD_class])

#AD_labels = np.array([0 for _ in range(len(AD_scans))])
CN_labels = np.array([0 for _ in range(len(CN_scans))])
MCI_labels = np.array([1 for _ in range(len(MCI_scans))])
MCI_AD_labels = np.array([2 for _ in range(len(MCI_AD_scans))])

train_set = np.concatenate((CN_scans, MCI_scans, MCI_AD_scans), axis=0)
test_set = np.concatenate((CN_labels, MCI_labels, MCI_AD_labels), axis=0)
from sklearn.model_selection import train_test_split
x_train, x_val = train_test_split(train_set, test_size=0.2, random_state=42)
y_train, y_val = train_test_split(test_set, test_size=0.2, random_state=42)

print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

def transform_images_dataset(data):
    #Binarize_images_dataset
    th=0.2
    upper=1
    lower=0
    data = np.where(data>th, upper, lower)
    #data_transform_channels
    data = data.reshape(data.shape[0], 1, 64, 64, 64)
    #data = np.stack((data,) * 3, axis=-1) #
    return(torch.as_tensor(data))

def one_hit_data(target):
    # Convert to torch Tensor
    target_tensor = torch.as_tensor(target)
    # Create one-hot encodings of labels
    one_hot = torch.nn.functional.one_hot(target_tensor.to(torch.int64), num_classes=num_classes)
    return(one_hot)

def random_crop(data):
    number = random.randrange(10 + 1)
    all_depth = data.shape[-1]
    all_width = data.shape[0]
    all_height = data.shape[1]
    size = 8
    for i in range(1,number + 1):
        depth_point = random.randrange(all_depth - size + 1)
        width_point = random.randrange(all_width - size + 1)
        height_point = random.randrange(all_height - size + 1)
        crop = tio.transforms.Crop(cropping=[width_point, width_point+size, height_point, height_point+size,
                                             depth_point, depth_point+size])
        data = crop(data)
    return data



y_train = one_hit_data(y_train)
y_val = one_hit_data(y_val)

x_train = transform_images_dataset(x_train)
x_test = transform_images_dataset(x_val)



train = torch.utils.data.TensorDataset(x_train.float(), y_train.long())
test = torch.utils.data.TensorDataset(x_test.float(), y_val.long())

batch_size = 32
train_loader = torch.utils.data.DataLoader(train, prefetch_factor=None
                                           , batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, prefetch_factor=None
                                           , batch_size=batch_size, shuffle=False)

print('train_loader',train_loader,'test_loader',test_loader)

from tqdm.auto import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

def accuracyFUNCTION (predicted, targets):
    c = 0
    for i in range(len(targets)):
        if (predicted[i] == targets[i]):
            #print("predict",predicted[i],"target",targets[i])
            c += 1
    accuracy = c / float(len(targets))
    print('accuracy = ', c, '/', len(targets))
    return accuracy

class CNN_classification_model(nn.Module):
    def __init__(self):
        super(CNN_classification_model, self).__init__()
        self.model = nn.Sequential(

            # Conv layer 1
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            # Conv layer 2
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            # Flatten
            nn.Flatten(),
            # Linear 1
            nn.Linear(2**3*7**3*128, 256),
            # Relu
            nn.ReLU(),
            # BatchNorm1d
            nn.BatchNorm1d(256),
            # Dropout
            nn.Dropout(p=0.3),
            # Linear 2
            nn.Linear(256, num_classes),
            # nn.Softmax(dim=1)

        )

    def forward(self, x):
        # Set 1
        #print('yyyy',x.shape)
        out = self.model(x)
        #out = F.softmax(out, dim=1)
        #out = torch.argmax(out,dim=1)
        #print("out", out)
        return out

num_epochs = 125
model = CNN_classification_model()
print(model)
error = nn.CrossEntropyLoss()
learning_r = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_r)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)

gpu = torch.cuda.is_available()
print("gpu =",gpu)
if gpu:
    model.cuda()

itr = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        if gpu:
            images, labels = images.cuda(), labels.cuda()

        train = Variable(images.view(images.shape))
        labels = Variable(labels)
        # Forward propagation / CNN_classification_model
        optimizer.zero_grad()
        outputs = model(train)
        # Calculate loss value / using cross entropy function
        #print("label",labels)
        labels = labels.argmax(-1)
        #print("label_max", labels)
        loss = error(outputs, labels)
        loss.backward()
        # Update parameters using SGD optimizer
        optimizer.step()

        # calculate the accuracy using test data
        itr += 1
        if itr % 50 == 0:
            # Prepare a list of correct results and a list of anticipated results.
            listLabels = []
            listpredicted = []
            # test_loader
            for images, labels in test_loader:
                if gpu:
                    images, labels = images.cuda(), labels.cuda()

                test = Variable(images.view(images.shape))
                # Forward propagation
                outputs = model(test)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]

                # used to convert the output to binary variables
                predicted = one_hit_data(predicted)
                # Create a list of predicted data
                predlist = []
                for i in range(len(predicted)):
                    p = int(torch.argmax(predicted[i]))
                    predlist.append(p)

                listLabels += (labels.argmax(-1).tolist())#labels.argmax(-1)
                listpredicted += (predlist)

                # calculate Accuracy
            accuracy = accuracyFUNCTION(listpredicted, listLabels)
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(itr, loss.data, accuracy))

            # store loss and accuracy. They'll be required to print the curve.
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)
    scheduler.step()

sns.set()
sns.set(rc={'figure.figsize':(12,7)}, font_scale=1)
plt.plot(accuracy_list,'b')
plt.plot(loss_list,'r')

plt.rcParams['figure.figsize'] = (7, 4)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training step:  Accuracy vs Loss ")
plt.legend(['Accuracy','Loss'])
plt.show()

predictionlist=[]
for i in range(len(outputs)):
    p = int(torch.argmax(outputs[i]))
    predictionlist.append(p)
labels1 = labels.argmax(-1).tolist()
labels1 = [str(a) for a in labels1]
predictionlist= [str(a) for a in predictionlist]
labelsLIST = ['0','1', '2']
cm = confusion_matrix(labels1, predictionlist, labels=labelsLIST)
ConfusionMatrixDisplay(cm).plot()
#   ******************** color of confusion matrix

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap=plt.cm.Blues); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels( ['0','1', '2']); ax.yaxis.set_ticklabels(['0','1', '2'])
plt.rcParams['figure.figsize'] = (8, 7)
plt.show()









