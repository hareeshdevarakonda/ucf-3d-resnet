import os
import numpy as np
#import cv2
import time
import h5py
import math
import keras
import random
from PIL import Image
from scipy import ndimage, misc
from sklearn.utils import shuffle

# from torchvision.datasets.folder import pil_loader

dataset = []
base_directory =  "/media/hareesh/hareesh/"
class_file = open(base_directory + 'ucfTrainTestlist/classInd.txt','r')
class_lines = class_file.readlines()
class_lines = [class_line.split(' ')[1].strip() for class_line in class_lines]
print (len(class_lines))
#print class_line

train_file = open(base_directory + 'ucfTrainTestlist/trainlist01.txt','r')
#print train_file
train_lines = train_file.readlines()
random.shuffle(train_lines)
#print (train_lines)
train_filenames = ['ucf101_jpeg/' + train_line.split(' ')[0] for train_line in train_lines]
#filename = ['ucf101_jpeg/' + line.split(' ')[0] for line in lines]
#print filenames
# filenames = [base_directory + filename for filename in filenames]
#print ("filenames",filenames)
# print (len(filenames))
train_file.close()

val_file = open(base_directory + 'ucfTrainTestlist/vallist01.txt','r')
val_lines = val_file.readlines()
random.shuffle(val_lines)
val_filenames = ['ucf101_jpeg/' + val_line.split(' ')[0] for val_line in val_lines]
#val_filenames = random.suffle(val_filenames)
val_file.close()

'''dataset1 = []
for filename in filenames :
	file_path = filename[:-4]
        print file_path
	n_frames_path = file_path+"/n_frames"
        print n_frames_path
       	n_frames_file = open(n_frames_path)
        #print (n_frames_file)
	n_frames = int(n_frames_file.readline())
	#print("this is number  "+str(n_frames))
        #n_frames = int(n_frames*0.1)
        #print n_frames
	frame_indices = []
        frame_indices = list(range(5,6))
        #frame_indices.append(random.randint(1, n_frames))      
        #frame_indices.append(random.randint(n_frames/10, n_frames/3))
        #frame_indices.append(random.randint(n_frames/3, n_frames*2/3))
        #frame_indices.append(random.randint(n_frames*2/3, n_frames))
        #random.randint(1, nb_clips/3)
        #print frame_indices
	video1 = video_loader(file_path,frame_indices)
        #print (len(video))
        #misc.imresize(image, (64, 64))
	dataset1.append(video1)	'''			 


#for class_line in lines:
    #print ("HAI")
#    if class_line in lines:
#        print "found the string"
#        print class_line

def image_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = misc.imresize(img, (224, 224))
            img = np.array(img, dtype=np.uint8)
            return img

def video_loader(video_dir_path, frame_indices):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        #image_path = ndimage.imread(image_path, mode="RGB")
        #image_path = misc.imresize(image, (64, 64))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video
'''
def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
'''
def dataGenerator(lines):
  while 1:
      #for i in range(num_class):
         # i= 0
         # while True:
         # class_wise_paths = []
         labels = []
         dataset = []
         for line in lines:
            # print(line)
            #if int(line.split(' ')[1]) == i + 1:
               path_name_1 = int(line.split(' ')[1])
               #print ("path_name_1",path_name_1)
               labels = []
               labels.append(path_name_1)
               path_name = "/media/hareesh/hareesh/ucf101_jpeg/" + line.split(' ')[0]
               # class_wise_paths.append(path_name)
               # print class_wise_paths
               path_name = path_name[:-4]
               # print path_name
               n_frames_path = path_name + "/n_frames"
               #print (n_frames_path)
               n_frames_file = open(n_frames_path)
               # print (n_frames_file)
               n_frames = int(n_frames_file.readline())
               #print(n_frames)
               n_frames = int(n_frames * 0.1)
               # print n_frames
               frame_indices = []
               #frame_indices = list(range(1, 3))
               frame_indices = list(range(1, 2))
               frame_indices.append(random.randint(2, n_frames-1))
               frame_indices.extend(list(range(n_frames-1, n_frames)))
               #frame_indices.append(random.randint(n_frames*2/3, n_frames))
               #print (frame_indices)
               video = video_loader(path_name, frame_indices)
               # print video
               dataset = []
               dataset.append(video)
               # print ("length of video",len(video))
               '''imglabels = []
               for i in range(len(video)):
                  print "labels",labels[0]
                  imglabels.append(labels[0])
               print "dataset-shape",(np.array(dataset).shape)    
   print "shape",(np.array(imglabels).shape)      
   yield np.array(dataset),np.array(imglabels)
               dataset =[]'''
               x_val = dataset
               #print(len(x_train))
               dataset = np.array(dataset)
               #print ("haiiiiiii",dataset.shape[0])
               del dataset
               # print x_train
               y_val = labels
               #print (len(y_train))
               del labels
               # batch_size =1
               x_val = np.array(x_val)
               #print ("byeeeee",x_train.shape[0])
               y_val = np.array(y_val)
               x_val,y_val = shuffle(x_val,y_val)
               #print ("////////",x_val)
               x_val = x_val.astype("float32")
               y_val = y_val.astype("float32")
               x_val /= 255
               y_val /= 255
               yield x_val, y_val
               x_val = []
               y_val = []

'''for index in dataGenerator(train_lines):
        print("length",len(index))
	#for value in index:
	    #print ("value",value)
        print("----------------")
        #print (index)'''

def val_dataGenerator(lines):
  while 1:
      #for i in range(num_class):
         # i= 0
         # while True:
         # class_wise_paths = []
         labels = []
         dataset = []
         for line in lines:
            # print(line)
            #if int(line.split(' ')[1]) == i + 1:
               path_name_1 = int(line.split(' ')[1])
               #print ("path_name_1",path_name_1)
               labels = []
               labels.append(path_name_1)
               path_name = "/media/hareesh/hareesh/ucf101_jpeg/" + line.split(' ')[0]
               # class_wise_paths.append(path_name)
               # print class_wise_paths
               path_name = path_name[:-4]
               # print path_name
               n_frames_path = path_name + "/n_frames"
               # print n_frames_path
               n_frames_file = open(n_frames_path)
               # print (n_frames_file)
               n_frames = int(n_frames_file.readline())
               # print(n_frames)
               n_frames = int(n_frames * 0.1)
               # print n_frames
               frame_indices = []
               #frame_indices = list(range(1, 3))
               frame_indices = list(range(1, 2))
               frame_indices.append(random.randint(2, n_frames-1))
               frame_indices.extend(list(range(n_frames-1, n_frames)))
               #frame_indices.append(random.randint(n_frames*2/3, n_frames))
               # print frame_indices
               video = video_loader(path_name, frame_indices)
               # print video
               print(np.array(video).shape);
               video = np.stack(video, axis = 2);
               print(np.array(video).shape);
               dataset = []
               dataset.append(video)
               # print ("length of video",len(video))
               '''imglabels = []
               for i in range(len(video)):
                  print "labels",labels[0]
                  imglabels.append(labels[0])
               print "dataset-shape",(np.array(dataset).shape)    
   print "shape",(np.array(imglabels).shape)      
   yield np.array(dataset),np.array(imglabels)
               dataset =[]'''
               x_train = video
               #print(len(x_train))
               dataset = np.array(dataset)
               #print ("haiiiiiii",dataset.shape[0])
               del dataset
               # print x_train
               y_train = labels
               #print (len(y_train))
               del labels
               # batch_size =1
               x_train = np.array(x_train)
               #print ("byeeeee",x_train.shape[0])
               y_train = np.array(y_train)
               x_train,y_train = shuffle(x_train,y_train)
               #print ("////////",x_val)
               x_train = x_train.astype("float32")
               y_train = y_train.astype("float32")
               x_train /= 255
               y_train /= 255
               yield x_train, y_train
               x_train = []
               y_train = []

from resnet3d import Resnet3DBuilder

model = Resnet3DBuilder.build_resnet_10((3, 224, 224, 3), 10)
model.summary()
#optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['categorical_accuracy'])
#model.fit_generator(dataGenerator(train_lines),steps_per_epoch=7058, epochs=2)

model.fit_generator(generator=dataGenerator(train_lines), validation_data=val_dataGenerator(val_lines), validation_steps=2472, steps_per_epoch=7058, epochs=5)

