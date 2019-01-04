from keras.applications import VGG16, VGG19

vgg16 = VGG16(include_top=False)
vgg16.summary()
for layer in vgg16.layers:
    print(layer.name)

vgg19 = VGG19(include_top=False)
vgg19.summary()
for layer in vgg19.layers:
    print(layer.name)
