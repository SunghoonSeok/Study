import os
import glob, numpy as np
from PIL import Image

# Image to npy
category_dir =  '../../data/lotte/train/'
categories = []
for i in range(0,1000) :
    i = "%d"%i
    categories.append(i)

classes = len(categories)
print(classes)

image_width = 128
image_height = 128
color = 3

pixels = image_height * image_width * color

X = []
y = []
for idx, product in enumerate(categories):
    
    label = [0 for i in range(classes)]
    label[idx] = 1

    image_dir = category_dir + "/" + product
    files = glob.glob(image_dir+"/*.jpg")

    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_width, image_height))
        data = np.asarray(img)

        X.append(data)
        y.append(label)

X = np.array(X)
y = np.array(y)
np.save("../../data/npy/lotte_train_x.npy", arr=X)
np.save("../../data/npy/lotte_train_y.npy", arr=y)
x = np.load("../../data/npy/lotte_train_x.npy",allow_pickle=True)
y = np.load("../../data/npy/lotte_train_y.npy",allow_pickle=True)

print(x.shape) # (48000, 128, 128, 3)
print(y.shape) # (72000, 1000)


img1=[]
for i in range(0,72000):
    filepath='../../data/lotte/test/%d.jpg'%i
    image=Image.open(filepath)
    image = image.convert('RGB')
    image = image.resize((128,128))
    image_data = asarray(image)
    img1.append(image_data)
    
np.save('../../data/npy/test.npy', arr=img1)
x_pred = np.load('../../data/npy/test.npy',allow_pickle=True)
print(x_pred.shape) # (72000, 128, 128, 3)