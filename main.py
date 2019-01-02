import time
import os
import numpy as np
import pandas as pd
from PIL import Image
from keras import backend
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from scipy.optimize import fmin_l_bfgs_b

# Specify image paths
c_image_path = './initial_images/french_horn.jpg'
s_image_path = './initial_images/starry_night.jpg'
o_image_directory = './horn_and_starry_night_output/'
directory = os.path.dirname(o_image_directory)
if not os.path.exists(directory):
    os.makedirs(directory)

# Image Processing
target_height = 512
target_width = 512
target_size = (target_height, target_width)

c_image_original = Image.open(c_image_path)
c_image_original_size = c_image_original.size
c_image = load_img(path=c_image_path, target_size=target_size)
c_image_arr = img_to_array(c_image)
c_image_arr = backend.variable(preprocess_input(np.expand_dims(c_image_arr, axis=0)), dtype='float32')

s_image = load_img(path=s_image_path, target_size=target_size)
s_image_arr = img_to_array(s_image)
s_image_arr = backend.variable(preprocess_input(np.expand_dims(s_image_arr, axis=0)), dtype='float32')

o_image_initial = np.random.randint(256, size=(target_width, target_height, 3)).astype('float64')
o_image_initial = preprocess_input(np.expand_dims(o_image_initial, axis=0))

o_image_placeholder = backend.placeholder(shape=(1, target_width, target_height, 3))

# Define the loss functions and other helper functions
def get_feature_reps(x, layer_names, model):
    feature_matrices = []
    for layer in layer_names:
        current_layer = model.get_layer(layer)
        feature_raw = current_layer.output
        feature_raw_shape = backend.shape(feature_raw).eval(session=tf_session)
        N_l = feature_raw_shape[-1]
        M_l = feature_raw_shape[1]*feature_raw_shape[2]
        feature_matrix = backend.reshape(feature_raw, (M_l, N_l))
        feature_matrix = backend.transpose(feature_matrix)
        feature_matrices.append(feature_matrix)
    return feature_matrices

def get_content_loss(F, P):
    content_loss = 0.5*backend.sum(backend.square(F - P))
    return content_loss

def get_gram_matrix(F):
    G = backend.dot(F, backend.transpose(F))
    return G

def get_style_loss(ws, Gs, As):
    style_loss = backend.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_l = backend.int_shape(G)[1]
        N_l = backend.int_shape(G)[0]
        G_gram = get_gram_matrix(G)
        A_gram = get_gram_matrix(A)
        style_loss += w*0.25*backend.sum(backend.square(G_gram - A_gram))/ (N_l**2 * M_l**2)
    return style_loss

def get_total_loss(o_image_placeholder, alpha=10.0, beta=10000.0):
    F = get_feature_reps(o_image_placeholder, layer_names=[c_layer_name], model=o_model)[0]
    Gs = get_feature_reps(o_image_placeholder, layer_names=s_layer_names, model=o_model)
    content_loss = get_content_loss(F, P)
    style_loss = get_style_loss(ws, Gs, As)
    total_lass = alpha * content_loss + beta * style_loss
    return total_lass

def calculate_loss(o_image_arr):
    """
    Calculate total loss using backend.function
    """
    if o_image_arr.shape != (1, target_width, target_width, 3):
        o_image_arr = o_image_arr.reshape((1, target_width, target_height, 3))
    loss_function = backend.function([o_model.input], [get_total_loss(o_model.input)])
    return loss_function([o_image_arr])[0].astype('float64')

def get_gradient(o_image_arr):
    """
    Calculate the gradient of the loss function with respect to the generated image
    """
    if o_image_arr.shape != (1, target_width, target_height, 3):
        o_image_arr = o_image_arr.reshape((1, target_width, target_height, 3))
    gradient_function = backend.function([o_model.input], backend.gradients(get_total_loss(o_model.input), [o_model.input]))
    gradient = gradient_function([o_image_arr])[0].flatten().astype('float64')
    return gradient

def postprocess_array(x):
    # Zero-center by mean pixel
    if x.shape != (target_width, target_height, 3):
        x = x.reshape((target_width, target_height, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

def save_image(x, image_number=None, title=None, target_size=c_image_original_size):
    x_image = Image.fromarray(x)
    x_image = x_image.resize(target_size)
    if image_number:
        image_path = o_image_directory + f'/image_at_iteration_{image_number}.jpg'
    elif title:
        image_path = o_image_directory + f'/{title}.jpg'
    else:
        image_path = o_image_directory + f'/output_image.jpg'
    x_image.save(image_path)
    return x_image

current_iteration = 0
def callback_image_save(xk):
    '''
    Callback function to save the image at certain iterations
    '''
    global current_iteration
    current_iteration += 1
    if current_iteration % 20 == 0:
        x_image = save_image(postprocess_array(xk), image_number=current_iteration)
        print('Image saved')

tf_session = backend.get_session()
c_model = VGG16(include_top=False, weights='imagenet', input_tensor=c_image_arr)
s_model = VGG16(include_top=False, weights='imagenet', input_tensor=s_image_arr)
o_model = VGG16(include_top=False, weights='imagenet', input_tensor=o_image_placeholder)

c_layer_name = 'block4_conv2'
s_layer_names = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
]

P = get_feature_reps(x=c_image_arr, layer_names=[c_layer_name], model=c_model)[0]
As = get_feature_reps(x=s_image_arr, layer_names=s_layer_names, model=s_model)
ws = np.ones(len(s_layer_names)) / float(len(s_layer_names))

iterations = 500
x_val = o_image_initial.flatten()

start = time.time()
x_output, f_minimum_val, info_dict = fmin_l_bfgs_b(func=calculate_loss, x0=x_val, fprime=get_gradient, maxiter=iterations, disp=True, callback=callback_image_save)
x_output = postprocess_array(x_output)
x_image = save_image(x_output, title='final_image')
print(f'Image saved')
end = time.time()
print(f'Time taken: {end - start}')
