import sys
import csv
import tflearn
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tflearn.data_utils import to_categorical



def linear_classifier(position_array, class_array, n_classes):
        # linear classifier
    with tf.Graph().as_default():
        # Build neural network
        net = tflearn.input_data(shape=[None, 2])
        # 'None' always has to be the first parameter in shape because it tells
        # tensor flow that the number of data points we have can be variable
        # and 2 for 2 input nodes (x and y coordinates)

        net = tflearn.fully_connected(net, n_classes, activation='softmax') # layer with 4 nodes and softmax
        net = tflearn.regression(net, loss='categorical_crossentropy') #regression with categorical_crossentropy

        # Define model
        model = tflearn.DNN(net)
        new_class_array = np.zeros((len(class_array), 4))
        index = 0

        #change to be 4 dimensional
        for x in class_array:
            if x == 0:
                new_class_array[index] = [1,0,0,0]
            elif x == 1:
                new_class_array[index] = [0,1,0,0]
            elif x == 2:
                 new_class_array[index]= [0,0,1,0]
            elif x == 3:
                new_class_array[index] = [0,0,0,1]
            index +=1 

        # Start training (apply gradient descent algorithm)
        model.fit(position_array, new_class_array, n_epoch=10, batch_size=10, show_metric=True, snapshot_step=1)
        return position_array, new_class_array, model


def non_linear_classifier(position_array, class_array, n_classes):
    with tf.Graph().as_default():
        
            # Build neural network
        net = tflearn.input_data(shape=[None, 2])
        # 'None' always has to be the first parameter in shape because it tells
        # tensor flow that the number of data points we have can be variable
        # and 2 for 2 input nodes (x and y coordinates)
        #sgd = tflearn.optimizers.SGD(learning_rate=2.0, lr_decay=0.5, decay_step=100)

        net = tflearn.fully_connected(net, 20000, activation='relu') # 20,0000 nodes
        net = tflearn.fully_connected(net, n_classes, activation='softmax') # layer with 4 nodes and softmax
        net = tflearn.regression(net, loss='categorical_crossentropy') #regression with categorical_crossentropy

        # Define model
        model = tflearn.DNN(net)
        new_class_array = np.zeros((len(class_array), 4))
        index = 0

        #change to be four dimensional
        for x in class_array:
            if x == 0:
                new_class_array[index] = [1,0,0,0]
            elif x == 1:
                new_class_array[index] = [0,1,0,0]
            elif x == 2:
                 new_class_array[index]= [0,0,1,0]
            elif x == 3:
                new_class_array[index] = [0,0,0,1]
            index +=1 

        # Start training (apply gradient descent algorithm)
        model.fit(position_array, new_class_array, n_epoch=10, batch_size=10, show_metric=True, snapshot_step=1)
        return position_array, new_class_array, model


def plot_spiral_and_predicted_class(position_array, class_array, model, output_file_name, title):
    h = 0.02
    x_min, x_max = position_array[:, 0].min() - 1, position_array[:, 0].max() + 1
    y_min, y_max = position_array[:, 1].min() - 1, position_array[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    z = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
    z = z.reshape(xx.shape)
    plt.close('all')
    fig = plt.figure()
    plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(position_array[:, 0], position_array[:, 1], c=class_array, s=40, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title(title)
    fig.savefig(output_file_name)


def plot_spiral(position_array, class_array, output_file_name):
    
    fig = plt.figure()
    plt.scatter(position_array[:, 0], position_array[:, 1], c=class_array, s=40, cmap=plt.cm.coolwarm)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    fig.savefig(output_file_name)


def get_accuracy(position_array, class_array, model):
    return np.mean(class_array == np.argmax(model.predict(position_array), axis=1))


def read_csv(path_to_file):
   
    position = []
    classification = []
    with open(path_to_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None) # skip the header

        for row in reader:
            position.append(np.array([float(row[0]), float(row[1])]))
            classification.append(float(row[2]))

    return np.array(position), np.array(classification, dtype='uint8')

def main():
    
    csv_file_name = ''
    do_linear = False
    
    if len(sys.argv) == 3:
        csv_file_name = sys.argv[1]
        do_linear = bool(int(sys.argv[2]))
    
    data = open(csv_file_name)

    #ignore title
    first_line = data.readline()
    
    #read the remaining data within the file
    body = data.readlines()
    length = len(body)
    position_array = np.zeros((length, 2))
    class_array = np.zeros((length, 1))
    
    index = 0
    for line in body:
        values = line.strip().split(',')
        position_array[index][0] = values[0]
        position_array[index][1] = values[1]
        class_array[index] = values[2]
        index += 1
    
    if do_linear == False:
        #plot_spiral(position_array, class_array, "spiral.png")
        linear_position_array, linear_class_array, linear_model = linear_classifier(position_array, class_array, 4)
        plot_spiral_and_predicted_class(linear_position_array, linear_class_array, linear_model, "linear.png", "Linear Classification Results")

    else:
        nonlinear_position_array, nonlinear_class_array, nonlinear_model = non_linear_classifier(position_array, class_array, 4)
        plot_spiral_and_predicted_class(nonlinear_position_array, nonlinear_class_array, nonlinear_model, "nonlinear.png", "Nonlinear Classification Results")



if __name__ == '__main__':
    main()
