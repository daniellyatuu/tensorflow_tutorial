from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np


class MyFunc(View):
    def get(self, request):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train_data = x_train[0]
        # plt.imshow(data1, cmap=plt.cm.binary)
        # plt.show()
        # y_train_data = y_train[0]

        # x_train = tf.keras.utils.normalize(x_train, axis=1)
        # x_test = tf.keras.utils.normalize(x_test, axis=1)
        # # print(x_train[0])
        # # plt.imshow(x_train[0], cmap=plt.cm.binary)
        # # plt.show()

        # model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.Flatten())

        # model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

        # model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        # model.compile(
        #     optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # model.fit(x_train, y_train, epochs=3)
        # val_loss, val_acc = model.evaluate(x_test, y_test)
        # model.save('epic_num_reader.model')

        new_model = tf.keras.models.load_model('epic_num_reader.model')
        predictions = new_model.predict(x_test)
        # print(predictions)
        # print(np.argmax(predictions[1]))

        plt.imshow(x_test[0], cmap=plt.cm.binary)
        plt.show()

        return HttpResponse('version')
