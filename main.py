import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

train_new_model = True

if train_new_model:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data() 

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1) 


    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) 

    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) 


    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    model.save('MNISTify.keras')
else:
    model = tf.keras.models.load_model('MNISTify.keras')

image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1
