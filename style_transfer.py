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

class StyleTransfer:
    def __init__(self):
        # Specify image paths
        self.c_image_path = './initial_images/french_horn.jpg'
        self.s_image_path = './initial_images/starry_night.jpg'
        self.o_image_directory = './horn_and_starry_night_output/'
        directory = os.path.dirname(self.o_image_directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Specify weights of content (alpha) and style (beta) loss
        self.alpha = 10.0
        self.beta = 10000.0

        # Create a text file that describes the parameters used in the script
        self.create_attributes_file()

        # Process the images
        self.process_images()

        # Tracks the iteration of the l_bfgs_b algorithm
        self.current_iteration = 0


    def create_attributes_file(self):
        with open(self.o_image_directory + 'attributes.txt', 'w') as f:
            f.write('Attributes of Style Transfer\n\n')
            f.write(f'Content image: {self.c_image_path[17:]}\n')
            f.write(f'Style image: {self.s_image_path[17:]}\n')
            f.write(f'Model used: VGG16\n')
            f.write(f'Alpha (content weight): {self.alpha}\n')
            f.write(f'Beta (style weight): {self.beta}')

    def process_images(self):
        # Image Processing
        self.target_height = 512
        self.target_width = 512
        self.target_size = (self.target_height, self.target_width)

        c_image_original = Image.open(self.c_image_path)
        self.c_image_original_size = c_image_original.size
        c_image = load_img(path=self.c_image_path, target_size=self.target_size)
        self.c_image_arr = img_to_array(c_image)
        self.c_image_arr = backend.variable(preprocess_input(np.expand_dims(self.c_image_arr, axis=0)), dtype='float32')

        s_image = load_img(path=self.s_image_path, target_size=self.target_size)
        self.s_image_arr = img_to_array(s_image)
        self.s_image_arr = backend.variable(preprocess_input(np.expand_dims(self.s_image_arr, axis=0)), dtype='float32')

        self.o_image_initial = np.random.randint(256, size=(self.target_width, self.target_height, 3)).astype('float64')
        self.o_image_initial = preprocess_input(np.expand_dims(self.o_image_initial, axis=0))
        self.o_image_placeholder = backend.placeholder(shape=(1, self.target_width, self.target_height, 3))

    def get_feature_reps(self, x, layer_names, model):
        feature_matrices = []
        for layer in layer_names:
            current_layer = model.get_layer(layer)
            feature_raw = current_layer.output
            feature_raw_shape = backend.shape(feature_raw).eval(session=self.tf_session)
            N_l = feature_raw_shape[-1]
            M_l = feature_raw_shape[1]*feature_raw_shape[2]
            feature_matrix = backend.reshape(feature_raw, (M_l, N_l))
            feature_matrix = backend.transpose(feature_matrix)
            feature_matrices.append(feature_matrix)
        return feature_matrices

    def get_content_loss(self, F, P):
        '''
        Calculuates the content loss using mean squared error
        '''
        content_loss = 0.5*backend.sum(backend.square(F - P))
        return content_loss

    def get_gram_matrix(self, F):
        '''
        Get the gram matrix for style loss function
        '''
        G = backend.dot(F, backend.transpose(F))
        return G

    def get_style_loss(self, ws, Gs, As):
        style_loss = backend.variable(0.)
        for w, G, A in zip(ws, Gs, As):
            M_l = backend.int_shape(G)[1]
            N_l = backend.int_shape(G)[0]
            G_gram = self.get_gram_matrix(G)
            A_gram = self.get_gram_matrix(A)
            style_loss += w*0.25*backend.sum(backend.square(G_gram - A_gram))/ (N_l**2 * M_l**2)
        return style_loss

    def get_total_loss(self, o_image_placeholder):
        F = self.get_feature_reps(o_image_placeholder, layer_names=[self.c_layer_name], model=self.o_model)[0]
        Gs = self.get_feature_reps(o_image_placeholder, layer_names=self.s_layer_names, model=self.o_model)
        content_loss = self.get_content_loss(F, self.P)
        style_loss = self.get_style_loss(self.ws, Gs, self.As)
        total_loss = self.alpha * content_loss + self.beta * style_loss
        return total_loss

    def calculate_loss(self, o_image_arr):
        '''
        Calculate total loss using backend.function
        '''
        if o_image_arr.shape != (1, self.target_width, self.target_width, 3):
            o_image_arr = o_image_arr.reshape((1, self.target_width, self.target_height, 3))
        loss_function = backend.function([self.o_model.input], [self.get_total_loss(self.o_model.input)])
        return loss_function([o_image_arr])[0].astype('float64')

    def get_gradient(self, o_image_arr):
        '''
        Calculate the gradient of the loss function with respect to the generated image using backend.function
        '''
        if o_image_arr.shape != (1, self.target_width, self.target_height, 3):
            o_image_arr = o_image_arr.reshape((1, self.target_width, self.target_height, 3))
        gradient_function = backend.function([self.o_model.input], backend.gradients(self.get_total_loss(self.o_model.input), [self.o_model.input]))
        gradient = gradient_function([o_image_arr])[0].flatten().astype('float64')
        return gradient

    def postprocess_array(self, x):
        # Zero-center by mean pixel
        if x.shape != (self.target_width, self.target_height, 3):
            x = x.reshape((self.target_width, self.target_height, 3))
        x[..., 0] += 103.939
        x[..., 1] += 116.779
        x[..., 2] += 123.68
        # 'BGR'->'RGB'
        x = x[..., ::-1]
        x = np.clip(x, 0, 255)
        x = x.astype('uint8')
        return x

    def save_image(self, x, image_number=None, title=None):
        x_image = Image.fromarray(x)
        x_image = x_image.resize(self.c_image_original_size)
        if image_number:
            image_path = self.o_image_directory + f'/image_at_iteration_{image_number}.jpg'
        elif title:
            image_path = self.o_image_directory + f'/{title}.jpg'
        else:
            image_path = self.o_image_directory + f'/output_image.jpg'
        x_image.save(image_path)

    def callback_image_save(self, xk):
        '''
        Callback function to save the image at certain iterations
        '''
        self.current_iteration += 1
        if self.current_iteration % 20 == 0 or self.current_iteration == 1:
            x_image = self.save_image(self.postprocess_array(xk), image_number=self.current_iteration)
            print('Image saved')

    def construct_image(self):
        self.tf_session = backend.get_session()
        self.c_model = VGG16(include_top=False, weights='imagenet', input_tensor=self.c_image_arr)
        self.s_model = VGG16(include_top=False, weights='imagenet', input_tensor=self.s_image_arr)
        self.o_model = VGG16(include_top=False, weights='imagenet', input_tensor=self.o_image_placeholder)

        self.c_layer_name = 'block4_conv2'
        self.s_layer_names = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
        ]

        self.P = self.get_feature_reps(x=self.c_image_arr, layer_names=[self.c_layer_name], model=self.c_model)[0]
        self.As = self.get_feature_reps(x=self.s_image_arr, layer_names=self.s_layer_names, model=self.s_model)
        self.ws = np.ones(len(self.s_layer_names)) / float(len(self.s_layer_names))

        iterations = 500
        x_val = self.o_image_initial.flatten()

        start = time.time()
        x_output, f_minimum_val, info_dict = fmin_l_bfgs_b(func=self.calculate_loss, x0=x_val, fprime=self.get_gradient, maxiter=iterations, disp=True, callback=self.callback_image_save)
        x_output = self.postprocess_array(x_output)
        self.save_image(x_output, title='final_image')
        print(f'Final image saved')
        end = time.time()
        print(f'Time taken: {end - start}')

if __name__ == '__main__':
    s = StyleTransfer()
    s.construct_image()
