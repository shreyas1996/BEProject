import tensorflow as tf
import time

def init_train():
    # Transfer learning (fine tuning) using VGG.net pre trained model
    # Image parameters
    image_size = 128
    train_batch_size = 52
    test_batch_size = 52

    Sequential = tf.keras.Sequential
    VGG16 = tf.keras.applications.VGG16
    models = tf.keras.models
    layers = tf.keras.layers
    optimizers = tf.keras.optimizers
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

    # Create VGG model (only Convolutional layers)
    vgg_conv = VGG16(weights = 'imagenet', include_top = False, 
                    input_shape = (image_size, image_size, 3)) 

    # Freeze the convolutional layers (make them non-trainable)
    # except last 4
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False
        
    # Check the trainable status of the layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

    # Create new fully connected model
    ft_model = Sequential([
        vgg_conv,
        layers.Flatten(),
        layers.Dense(units = 1024, activation = 'relu'),
        layers.Dropout(0.5),
        layers.Dense(units = 1, activation = 'sigmoid')
    ])

    # model summary
    ft_model.summary()

    # Pre process and augment the images
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
                optimizer = optimizers.SGD(learning_rate = 0.0001, decay = 1e-6,
                                            momentum = 0.9, 
                                            nesterov = True))

    # optimizer = optimizers.RMSprop(lr = 1e-4)

    start_time = time.time()
    # Train the model
    history = ft_model.fit(training_set,
                                    epochs = 50,
                                    steps_per_epoch = training_set.samples/training_set.batch_size,
                                    validation_data = test_set,
                                    validation_steps = test_set.samples/test_set.batch_size,
                                    verbose = 1)

    end_time = time.time()
    train_time =end_time - start_time

    # Save the model
    ft_model.save('refined_ft_model_final.h5')

    print("training completed in ", train_time, "seconds")
    print(history)

    return "Training completed in " + str(train_time) + "seconds"