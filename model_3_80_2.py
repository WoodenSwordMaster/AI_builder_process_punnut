import os
import nibabel as nib
from scipy import ndimage
import numpy as np
from numpy import random
import torch
import matplotlib.pyplot as plt
import torchio as tio
import cv2

data_path = r"C:\Users\punnut\Downloads\final_dataset1_preprocessed3+AD_traintest_80"
AD_class_train = []
CN_class_train = []
MCI_class_train = []
MCI_AD_class_train = []
AD_class_val = []
CN_class_val = []
MCI_class_val = []
MCI_AD_class_val = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".nii"):
            pathList = root.split(os.sep)
            for i in pathList:
                if i == "train":
                    for j in pathList:
                        if j == "AD":
                            AD_class_train.append(os.path.join(root, file))
                            # print(os.path.join(root,file))
                        elif j == "CN":
                            CN_class_train.append(os.path.join(root, file))
                            # print(os.path.join(root, file))
                        elif j == "MCI":
                            MCI_class_train.append(os.path.join(root, file))
                            # print(os.path.join(root, file))
                        elif j == "MCI_AD":
                            MCI_AD_class_train.append(os.path.join(root, file))
                            # print(os.path.join(root, file))
                elif i == "test":
                    for j in pathList:
                        if j == "AD":
                            AD_class_val.append(os.path.join(root, file))
                            # print(os.path.join(root,file))
                        elif j == "CN":
                            CN_class_val.append(os.path.join(root, file))
                            # print(os.path.join(root, file))
                        elif j == "MCI":
                            MCI_class_val.append(os.path.join(root, file))
                            # print(os.path.join(root, file))
                        elif j == "MCI_AD":
                            MCI_AD_class_val.append(os.path.join(root, file))
                            # print(os.path.join(root, file))


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    print(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

# def normalize2(volume):
#     # Normalize the volume to zero mean and unit variance
#     volume = (volume - np.mean(volume)) / np.std(volume)  # <-- Added normalization
#     volume = volume.astype("float32")
#     return volume
#
# def resize_volume(img):
#     """Resize across z-axis"""
#     # Set the desired depth
#     desired_depth = 96
#     desired_width = 96
#     desired_height = 96
#     # Get current depth
#     current_depth = img.shape[-1]
#     current_width = img.shape[0]
#     current_height = img.shape[1]
#     # Compute depth factor
#     depth = current_depth / desired_depth
#     width = current_width / desired_width
#     height = current_height / desired_height
#     depth_factor = 1 / depth
#     width_factor = 1 / width
#     height_factor = 1 / height
#     # Rotate
#     img = ndimage.rotate(img, 90, reshape=False)
#     # Resize across z-axis
#     img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
#     return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    #volume = normalize2(volume)
    # Resize width, height and depth
    #volume = resize_volume(volume)
    return volume

num_classes = 4
AD_scans_train = np.array([process_scan(path) for path in AD_class_train])
CN_scans_train = np.array([process_scan(path) for path in CN_class_train])
MCI_scans_train = np.array([process_scan(path) for path in MCI_class_train])
MCI_AD_scans_train = np.array([process_scan(path) for path in MCI_AD_class_train])
AD_scans_val = np.array([process_scan(path) for path in AD_class_val])
CN_scans_val = np.array([process_scan(path) for path in CN_class_val])
MCI_scans_val = np.array([process_scan(path) for path in MCI_class_val])
MCI_AD_scans_val = np.array([process_scan(path) for path in MCI_AD_class_val])

AD_labels_train = np.array([0 for _ in range(len(AD_scans_train))])
CN_labels_train = np.array([1 for _ in range(len(CN_scans_train))])
MCI_labels_train = np.array([2 for _ in range(len(MCI_scans_train))])
MCI_AD_labels_train = np.array([3 for _ in range(len(MCI_AD_scans_train))])
AD_labels_val = np.array([0 for _ in range(len(AD_scans_val))])
CN_labels_val = np.array([1 for _ in range(len(CN_scans_val))])
MCI_labels_val = np.array([2 for _ in range(len(MCI_scans_val))])
MCI_AD_labels_val = np.array([3 for _ in range(len(MCI_AD_scans_val))])

x_train = np.concatenate((CN_scans_train, MCI_scans_train, MCI_AD_scans_train), axis=0)
x_val = np.concatenate((CN_scans_val, MCI_scans_val, MCI_AD_scans_val), axis=0)
y_train = np.concatenate((CN_labels_train, MCI_labels_train, MCI_AD_labels_train), axis=0)
y_val = np.concatenate((CN_labels_val, MCI_labels_val, MCI_AD_labels_val), axis=0)

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
    data = data.reshape(data.shape[0], 1, 80, 80, 80)
    #data = np.stack((data,) * 3, axis=-1) #
    return(torch.as_tensor(data))

def one_hit_data(target):
    # Convert to torch Tensor
    target_tensor = torch.as_tensor(target)
    # Create one-hot encodings of labels
    one_hot = torch.nn.functional.one_hot(target_tensor.to(torch.int64), num_classes=num_classes)
    return(one_hot)


y_train = one_hit_data(y_train)
y_val = one_hit_data(y_val)

x_train = transform_images_dataset(x_train)
x_val = transform_images_dataset(x_val)


train = torch.utils.data.TensorDataset(x_train.float(), y_train.long())
val = torch.utils.data.TensorDataset(x_val.float(), y_val.long())

batch_size = 32
train_loader = torch.utils.data.DataLoader(train, prefetch_factor=None
                                           , batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, prefetch_factor=None
                                           , batch_size=batch_size, shuffle=False)

print('train_loader',train_loader,'test_loader',val_loader)

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
        #self.model = nn.Sequential(

        # Conv layer 1
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d((2, 2, 2))

        # Conv layer 2
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=0)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool3d((2, 2, 2))

        # self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=0)
        # self.relu4 = nn.ReLU()
        # self.maxpool4 = nn.MaxPool3d((2, 2, 2))

        # Flatten
        self.flat = nn.Flatten()
        # Linear 1
        self.linear1 = nn.Linear(8**3*128, 256)
        # Relu
        self.relu3 = nn.ReLU()
        # BatchNorm1d
        self.norm = nn.BatchNorm1d(256)
        # Dropout
        self.drop = nn.Dropout(p=0.3)
        # Linear 2
        self.linear2 = nn.Linear(256, 4) #num_classes
        # nn.Softmax(dim=1)

        #)

    def forward(self, x):
        #out = self.model(x)
        #out = torch.argmax(out,dim=1)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = self.maxpool4(x)
        print("maxpool4",x.shape)

        x = self.flat(x)
        print("flat",x.shape)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.norm(x)
        x = self.drop(x)
        out = self.linear2(x)

        #print("out", out)
        return out

num_epochs = 10  #50
model = CNN_classification_model()
print(model)
error = nn.CrossEntropyLoss()
learning_r = 0.01
optimizer = torch.optim.AdamW(nn.ParameterList(model.parameters()), lr=learning_r)
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
        model.train()
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
            model.eval()
            val_loss = 0
            count = 0
            for images, labels in val_loader:
                if gpu:
                    images, labels = images.cuda(), labels.cuda()

                test = Variable(images.view(images.shape))
                # Forward propagation
                outputs2 = model(test)

                # Get predictions from the maximum value
                predicted = torch.max(outputs2.data, 1)[1]

                # used to convert the output to binary variables
                predicted = one_hit_data(predicted)
                # Create a list of predicted data
                predlist = []
                for i in range(len(predicted)):
                    p = int(torch.argmax(predicted[i]))
                    predlist.append(p)
                listLabels += (labels.argmax(-1).tolist())
                listpredicted += (predlist)

                labels = labels.argmax(-1)
                loss2 = error(outputs2, labels)
                count += count
                # print("count",count)
                # print("loss2", loss2.item() * images.size(0))
                val_loss += loss2.item() * images.size(0)
                #print("val_loss", val_loss)

                # calculate Accuracy
            accuracy = accuracyFUNCTION(listpredicted, listLabels)
            val_loss = val_loss / len(val_loader.dataset)
            # print("loss", loss.data())
            # print("all_val_loss", val_loss)
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(itr, val_loss, accuracy))

            # store loss and accuracy. They'll be required to print the curve.
            loss_list.append(val_loss)
            accuracy_list.append(accuracy)
        scheduler.step()

torch.save(model.state_dict(), r'C:\Users\punnut\Downloads\saved_model2_80.pth')
from sklearn.metrics import classification_report
model.eval()
confusion_predict = []
confusion_label = []
listLabels = []
listpredicted = []
for images, labels in tqdm(val_loader):
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
        confusion_predict.append(p)

    label_list = labels.argmax(-1).tolist()
    for i in range(len(label_list)):
        p = int(label_list[i])
        confusion_label.append(p)

    listLabels += (label_list)
    listpredicted += (predlist)
print(classification_report(listLabels, listpredicted))

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

# predictionlist=[]
# for i in range(len(outputs)):
#     p = int(torch.argmax(outputs[i]))
#     predictionlist.append(p)
# labels1 = labels.argmax(-1).tolist()
labels1 = [str(a) for a in confusion_label]
predictionlist= [str(a) for a in confusion_predict]
labelsLIST = ['0','1', '2', '3']
cm = confusion_matrix(labels1, predictionlist, labels=labelsLIST)
ConfusionMatrixDisplay(cm).plot()
#   ******************** color of confusion matrix

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap=plt.cm.Blues); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels( ['0','1', '2', '3']);
ax.yaxis.set_ticklabels(['0','1', '2', '3'])
plt.rcParams['figure.figsize'] = (8, 7)
plt.show()









