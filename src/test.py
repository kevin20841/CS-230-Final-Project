# example of using saved cyclegan models for image translation
from tensorflow.keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from tensorflow_addons.layers import InstanceNormalization
import os
import tensorflow as tf
import numpy as np


# load and prepare training images
def get_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.rgb_to_hsv(image)
    image = tf.multiply(tf.subtract(image, 0.5),2)
    return image

def save_image(images, filename):
    count = 0
    os.makedirs(filename, exist_ok=True)
    for image in images:
        
        image = tf.divide(tf.add(image, 1), 2)
        image = tf.image.hsv_to_rgb(image)
        image = tf.multiply(image, 255)

        
        
        image = tf.cast(tf.image.resize(image, [250, 375]), tf.uint8)
        image = tf.image.encode_png(image)
        f = open(os.path.join(filename, str(count) + ".png"), "wb+")
        f.write(image.numpy())
        f.close()
        count +=1 
     


# select a random sample of images from the dataset

def select_sample(path, names, n_samples):
    names = np.array(names)
    random_classes = np.random.randint(0, len(names), n_samples)
    classes = names[random_classes]
    directionsA = np.random.randint(0, 24, n_samples)
    directionsB = directionsA + 1

    directionsA = directionsA.astype(str)
    directionsB = directionsB.astype(str)


    X1 = []
    X2 = []
    for i in range(len(classes)):
        X1.append(path + classes[i] + "/dir_" + directionsA[i] + "_mip2.jpg")
        X2.append(path + classes[i] + "/dir_" + directionsB[i] + "_mip2.jpg")

    X1t = tf.map_fn(get_image, tf.convert_to_tensor(X1), dtype=tf.float32)
    X2t = tf.map_fn(get_image, tf.convert_to_tensor(X2), dtype=tf.float32)
    
    return X1t, X2t

# plot the image, the translation, and the reconstruction


def load_dataset(dataset_path):
    print("Loading files from: "+ os.path.join(dataset_path, "*/*.jpg"))

    names = [directory for directory in os.listdir(dataset_path) if os.path.isdir(dataset_path+directory)]

    return names
    

# load dataset
path = "../data/test_set/"
names = load_dataset(path)
n_samples = 10
model_names = ["000490","000980", "001470", "001960", "002450", "002940", "003430", "003920" ]
A_real, B_real = select_sample(path, names, n_samples)
save_image(A_real, "./imghsv/A_real")
save_image(B_real, "./imghsv/B_real")

for model in model_names:
    # load the models
    cust = {'InstanceNormalization': InstanceNormalization}
    model_AtoB = load_model("./g_model_AtoB_" + model, cust, compile=False)
    model_BtoA = load_model("./g_model_BtoA_" + model, cust, compile=False)
    # plot A->B->A
    B_generated = model_AtoB.predict(A_real)
    A_reconstructed = model_BtoA.predict(B_generated)
    
    A_generated = model_BtoA.predict(B_real)
    B_reconstructed = model_AtoB(A_generated)

    save_image(B_generated, "./imghsv/B_generated/"+model)
    save_image(A_reconstructed, "./imghsv/A_reconstructed/" + model)
    save_image(A_generated, "./imghsv/A_generated/"+model)
    save_image(B_reconstructed, "./imghsv/B_reconstructed/"+model)



