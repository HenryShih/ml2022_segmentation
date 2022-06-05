import cv2
import matplotlib.pyplot as plt
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import glob
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import tensorflow as tf
import timeit

from pathlib import Path
from tensorflow.python.framework.graph_util import convert_variables_to_constants



class DataLoaderSegmentation(data.Dataset):
    def __init__(self, input_path,label_path,label_name='',transform=None):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(input_path,'*.jpg'))
        self.mask_files = []
        self.transforms = transform
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(label_path,os.path.basename(img_path).split('.')[0]+label_name+'.png'))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = cv2.imread(img_path)
            label = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            label = F.one_hot(torch.from_numpy(label).to(torch.int64),6)
            datalabel = np.concatenate((data,label),axis=2)
            datalabel = np.transpose(datalabel,[2,0,1])
            if self.transforms!=None:
              datalabel = self.transforms(torch.from_numpy(datalabel).float())
            datalabel = np.transpose(datalabel,[1,2,0])
            data  = datalabel[:,:,0:3]
            label = datalabel[:,:,3:9]
            return data,label

    def __len__(self):
        return len(self.img_files)


#prepare training/validation datasets
batch_size = 25

input_path='../ICME2022_Training_Dataset/images' #720/1280
label_path='../ICME2022_Training_Dataset/labels/class_labels'
dataset = DataLoaderSegmentation(input_path,label_path,'_lane_line_label_id',transforms.Resize(size=(720,1280)))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


input_path='../ICME2022_Training_Dataset/images_real_world' #1080/1920
label_path='../ICME2022_Training_Dataset/labels_real_world'
dataset_real = DataLoaderSegmentation(input_path,label_path,'',transforms.Resize(size=(1080,1920)))
dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=batch_size, shuffle=True)


#design the network
inputs = tf.placeholder(tf.float32,shape=(None,None, None, 3))
y_ = tf.placeholder(tf.float32, [None,None, None,6])
x = tf.image.resize_images(inputs, (256, 256))
x = x/255.0
y = tf.image.resize_images(y_, (256, 256))
ch= 15 #channel
depth= 7 #depth
xn = []
b=tf.Variable(0.0)
x= tf.layers.conv2d(x, ch, 3, 1, 'same') #FIXME
x= tf.layers.batch_normalization(x) #FIXME
x= tf.nn.relu(x) #FIXME
for i in range(depth):
    xn.append(x)
    x = tf.layers.conv2d(x,ch*(2**(i+1)),3,1,'same')
    x = tf.layers.batch_normalization(x,center=False,scale=False)+b
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x,ch*(2**(i+1)),3,1,'same')
    x = tf.layers.batch_normalization(x,center=False,scale=False)+b
    x = tf.nn.relu(x)
    if i < depth-1:
        x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],'SAME')
    print(i, 'layer', x.shape)
for i in range(depth):
    if i>0:
        x = tf.keras.layers.UpSampling2D((2,2))(x)
    x = tf.layers.conv2d(x,ch*(2**(depth-i-1)),3,1,'same')+xn[-i-1]
    x = tf.layers.batch_normalization(x,center=False,scale=False)+b
    x = tf.nn.relu(x)
out = tf.layers.conv2d(x,6,3,1,'same')
outputs = out
outputs = tf.image.resize_images(outputs, (1080, 1920 ))
outputs = tf.argmax(outputs,-1)


#set the hyperparameters
loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=out,labels=y)
loss=tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001)
train = optimizer.minimize(loss+0.0005*b)
saver=tf.train.Saver()
init = tf.global_variables_initializer()


#session run and restore checkpoint
sess = tf.Session()
sess.run(init)
# saver.restore(sess, '/content/MediaTek_IEE5725_Machine_Learning_Lab3/model/')



#Set training Epochs, Print the training logs and Save your Checkpoint
num_epochs = 1
for epoch in range(num_epochs):
    start_time = timeit.default_timer()
    for i, data in enumerate(dataloader, 0):
        input = data[0].numpy()
        label = data[1].numpy()
        sess.run(train,feed_dict={inputs: input, y_: label})
        if i % 10 == 0:
            print("[%d/%d][%s/%d] loss: %.4f b: %.4f "\
                %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), sess.run(loss,feed_dict={inputs: input, y_: label}),sess.run(b)) )
        if i%300==0:
            # print('checkpoint saved')
            for i, data in enumerate(dataloader_real, 0):
                input = data[0].numpy()
                label = data[1].numpy()
                sess.run(train,feed_dict={inputs: input, y_: label})
            # saver.save(sess, 'drive/MyDrive/MediaTek_IEE5725_Machine_Learning_Lab3/model/')
    saver.save(sess, './checkpoints/checkpoint_epoch', gobal_step=epoch+1)
    print('checkpoint saved')
    end_time = timeit.default_timer()
    print(f'end of epoch {epoch+1}, spending {int((end_time-start_time)/60)} minites')










