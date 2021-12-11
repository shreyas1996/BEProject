# Transfer learning (fine tuning) using VGG.net pre trained model
import time
# Image parameters
image_size = 128
train_batch_size = 52
test_batch_size = 52

# Create VGG model (only Convolutional layers)
from keras.applications import VGG16

vgg_conv = VGG16(weights = 'imagenet', include_top = False, 
                 input_shape = (image_size, image_size, 3)) 

# Freeze the convolutional layers (make them non-trainable)
# except last 4
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
    
# Check the trainable status of the layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
    
# plot vgg model    
'''from keras.utils import plot_model
plot_model(vgg_conv, to_file = 'vgg_conv.png')'''

# Create new fully connected model
from keras import models
from keras import layers
from keras import optimizers

ft_model = models.Sequential() 

# Add pre-trained model
ft_model.add(vgg_conv)

# Add new layers
ft_model.add(layers.Flatten())
ft_model.add(layers.Dense(units = 1024, activation = 'relu'))
ft_model.add(layers.Dropout(0.5))
ft_model.add(layers.Dense(units = 1, activation = 'sigmoid'))

# model summary
ft_model.summary()

# plot fine-tuned model
#plot_model(ft_model, to_file = 'ft_model.png')

# Pre process and augment the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(horizontal_flip = True,
                                   rescale = 1./255, 
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   fill_mode = 'nearest',
                                   rotation_range = 20)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(directory = 'dataset/training',
                                                 target_size = (image_size, image_size),
                                                 class_mode = 'binary', 
                                                 batch_size = train_batch_size)

test_set = test_datagen.flow_from_directory(directory = 'dataset/testing',
                                            target_size = (image_size, image_size),
                                            class_mode = 'binary',
                                            batch_size = test_batch_size,
                                            shuffle = False)

# Compile the model
ft_model.compile(loss = 'binary_crossentropy', 
              metrics = ['accuracy'],
              optimizer = optimizers.SGD(lr = 0.001, decay = 1e-6,
                                         momentum = 0.9, 
                                         nesterov = True))

# optimizer = optimizers.RMSprop(lr = 1e-4)

start_time = time.time()
# Train the model
history = ft_model.fit_generator(training_set,
                                 epochs = 50,
                                 steps_per_epoch = training_set.samples/training_set.batch_size,
                                 validation_data = test_set,
                                 validation_steps = test_set.samples/test_set.batch_size,
                                 verbose = 1)

end_time = time.time()
train_time =end_time - start_time
# Save the model
ft_model.save('ft_model4.h5')

print("training completed in ", train_time, "seconds")

print(history)