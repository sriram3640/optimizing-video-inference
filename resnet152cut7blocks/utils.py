import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model


def plot(model, layer_name, test_data):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test_data)
    fig = plt.figure(figsize=(8,8))
    for i in range(5):
        ax = fig.add_subplot(6,6,i+1)
        ax.imshow(intermediate_output[0,:,:,i])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    plt


def get_names(model):
    for layer in model.layers:
        print(layer.name)


def plot_normal(img):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(6, 6, 3)
    ax.imshow(img)

def batch_set_weights(model1, n_layer1, model2, n_layer2):
    for i, j in zip(range(n_layer1, len(model1.layers)),range(n_layer2, len(model2.layers))):
        model1.layers[i].set_weights(model2.layers[j].get_weights())

def batch_compile(models):
    for model in models:
        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

