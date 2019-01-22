import os
import imageio

o_image_directory = './output/llama_and_starry_night/'

images = []
counter = 0
for filename in os.listdir(o_image_directory):
    if os.path.splitext(filename)[1] == '.jpg' and counter < 13:
        images.append(imageio.imread(o_image_directory + filename))
        counter += 1
imageio.mimsave(o_image_directory + 'collected_images.gif', images, duration=0.3)
