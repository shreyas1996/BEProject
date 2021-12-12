#!C:\Users\krishnamoothy\AppData\Local\conda\conda\envs\tensorflow\python
print("Content-Type: text/html")
print()

# Make single prediction
import numpy as np
from keras.preprocessing import image

'''import os
os.chdir('C:/xampp/project')'''

from keras.models import load_model
cnn = load_model('cnn_bs.h5')

# import sys
# filepath=sys.argv[1]
test_image = image.load_img('1.jpg', 
                            target_size = (128, 128))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = cnn.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'Leukemia'
else:
    prediction = 'Healthy'

print(prediction)

# Save the model
# cnn.save('cnn_bs.h5')