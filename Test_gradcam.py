import numpy as np
import base64
import flask
import io
import requests
from keras.backend import tensorflow_backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model
import tensorflow as tf
#from tensorflow import keras
import keras
import cv2
from flask import send_file
from PIL import Image
from keras_efficientnets import EfficientNetB5
from tensorflow.keras.preprocessing import image
import threading
import multiprocessing as mp
from multiprocessing import Pool,Process
import ray
#tf.enable_eager_execution()


mp.set_start_method('fork')

IMG_SIZE = 224
tol = 7
sigmaX = 10
H , W = IMG_SIZE*4,IMG_SIZE*4
all_grads = []
my_cam = []
base_map = np.zeros((56,56),dtype=int)

app = flask.Flask(__name__)


graph2 = tf.Graph()
with graph2.as_default():
    global model2,session2
    session2 = tf.Session()
    with session2.as_default():
        K.set_session(session2)
        model2= None
        model2 = load_model("./intl10.h5", compile=True)

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x

def crop_image_from_gray(img):

    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def load_gauss(img):  # load_ben_color
    #img = cvtColor(array(Image.open(im_path)), COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return img

def keras_preprocessing(img, *p):
    # if p == 1:
    #     img = cvtColor(array(Image.open(im_path)), COLOR_BGR2RGB)

    img = load_gauss(img=img)
    gauss_img = img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target_size = (IMG_SIZE, IMG_SIZE)
    img = cv2.resize(img, target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img / 255, axis=0)
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    return img,gauss_img

def normalise_cam(cam):

    cam = np.maximum(cam, 0)
    cam_max = cam.max()
    if cam_max != 0:
        cam = cam / cam_max

    return cam

def get_all_activation_layer(activation_layer_list,model):

    global act_layers
    act_layers = []

    for activation_layer in activation_layer_list:

        print("Activation layer computed")
        act_layers.append(model.get_layer(activation_layer).output)

    #eturn act_layers

def get_conv_layer(conv_layer_name,model):

    global conv_output

    conv_output = model.get_layer(conv_layer_name).output

    #return conv_output


grads_list=[]
gradient_function_list = []
kk =[]

def gradcam(org_img):


    conv_layer = "conv2d_51"
    activation_layer_list = ["swish_38", "swish_39", "swish_40", "activation_14"]

    base_map = np.zeros((28, 28), dtype=int)
    res = 14*64
    base_map_max = base_map
    base_map_mean = base_map


    global img_str, buffer, my_str, gauss_encoded_img, gradcam_encoded_img, overlap_encoded_img,jetcam_max,jetcam_mean

    with session2.as_default(), graph2.as_default():


        if len(gradient_function_list)==0:


            get_conv_layer(conv_layer_name=conv_layer, model=model2)
            get_all_activation_layer(activation_layer_list=activation_layer_list, model=model2)

            for idx, act_output in enumerate(act_layers):
                grads = K.gradients(act_output, conv_output)[0]
                gradient_function_list.append(K.function([model2.input], [conv_output, grads]))

        img_array, gauss_img = keras_preprocessing(org_img)

        for idx, act_output in enumerate(act_layers):
            print("calculating gradient for  :", act_layers[idx], "  layer")

            output, grads_val = gradient_function_list[idx]([img_array])
            output, grads_val = output[0, :], grads_val[0, :, :, :]

            cam_max = np.dot(output, np.max(grads_val, axis=(0, 1)))
            cam_mean = np.dot(output, np.mean(grads_val, axis=(0, 1)))

            # cam_max = normalise_cam(cam_max)
            # cam_mean = normalise_cam(cam_mean)

            base_map_max = base_map_max + cam_max
            base_map_mean = base_map_mean + cam_mean

        base_map_mean = normalise_cam(base_map_mean)
        base_map_max = normalise_cam(base_map_max)

        cam_max = cv2.resize(base_map_max, (res,res), cv2.INTER_LINEAR)
        cam_max = normalise_cam(cam_max)

        cam_mean = cv2.resize(base_map_mean, (res, res), cv2.INTER_LINEAR)
        cam_mean = normalise_cam(cam_mean)

        jetcam_max = cv2.applyColorMap(np.uint8(255 * cam_max), cv2.COLORMAP_JET)
        jetcam_mean = cv2.applyColorMap(np.uint8(255 * cam_mean), cv2.COLORMAP_JET)

        cv2.imwrite('gauss.jpg', np.uint8(cv2.resize(gauss_img, (H, W))))

        cv2.imwrite('gradcam_max.jpg', np.uint8(jetcam_max))
        cv2.imwrite('overlap_max.jpg', 0.6 * np.uint8(cv2.resize(gauss_img, (H, W))) + 0.4 * np.uint8(cv2.resize(jetcam_max, (H, W))))

        cv2.imwrite('gradcam_mean.jpg', np.uint8(jetcam_mean))
        cv2.imwrite('overlap_mean.jpg', 0.6 * np.uint8(cv2.resize(gauss_img, (H, W))) + 0.4 * np.uint8(cv2.resize(jetcam_mean, (H, W))))

        return gauss_img,jetcam_max,jetcam_mean

img = cv2.imread('./test_images/DR_4.JPG')

gradcam(img)



# cv2.imshow("gradcam_max",cv2.imread('overlap_max.jpg'))
# cv2.imshow("gradcam_mean",cv2.imread('overlap_mean.jpg'))
# cv2.imshow("gauss",cv2.imread('gauss.jpg'))
# cv2.waitKey(0)




