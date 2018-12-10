import numpy as np 
# data analysis library used to read rows and columns in dataset
import pandas as pd 
# splits arrays and matrices into random train and test subsets
from sklearn.model_selection import train_test_split 
# neural network library
import keras
# linear stack of layers
from keras.models import Sequential 
# optimization for gradient descent
from keras.optimizers import Adam 
# save model with ModelCheckPoint
from keras.callbacks import ModelCheckpoint
# CNN Layers
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout 
# helper methods
from helper import INPUT_SHAPE, batch_generator 

# read files
import os 

# seed train_test_split for training and validation data
np.random.seed(0)

'''
Written by:
Brian Powell @BriianPowell
Justin Concepcion @JustinConcepcion
CECS 456: Machine Learning
Wenlu Zhang
'''

def loadData(data_dir, test_size):
    # reads CSV dataset into a dataframe = data_df 
    data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    # store camera images into input data, x
    # images include center, right, and left
    x = data_df[['center', 'left', 'right']].values
    # store the steering angle data as output, y
    y = data_df['steering'].values
    # splits training and testing sets, 20% of data set to test split
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=test_size, random_state=0) 

    return x_train, x_valid, y_train, y_valid


def buildModel(drop_out):
    '''
    Smaller implementation of a VGGNet 
    First Layer is image normalization as suggested by NVIDIA
    3 Convolution Layers
    Each is followed by Max Pool
    Drop Out is 0.5
    3 Fully Connected layers to get one final output
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE)) #normalization
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dropout(drop_out))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    model.summary()

    return model

    '''
    NVIDIA's CNN Architecture
    First layer performs image normalization followed by 5 convolutional layers.
    Dropout is added to avoid overfitting.
    Finally, 3 fully connected layers leads to a final output steering command.
    The fully connected layers are designed to function as a controller for steering.
    
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE)) #normalization
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(drop_out))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    '''

def trainModel(model, data_dir, batch_size, samples_per_epoch, epochs, save_best_only, learning_rate, x_train, x_valid, y_train, y_valid):
    # ModelCheckPoint saves each model when save_best_only has been fullfilled
    checkpoint = ModelCheckpoint('lakeModel-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=save_best_only,
                                 mode='auto') #model's weights will be saved
	
    # mean squared error used to calculate loss
    # Adam optimization algorithm for stochastic gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

    # trains model on data generated batch by batch
    model.fit_generator(batch_generator(data_dir, x_train, y_train, batch_size, True),
                        samples_per_epoch,
                        epochs,
                        verbose=1,
                        callbacks=[checkpoint],
                        validation_data=batch_generator(data_dir, x_valid, y_valid, batch_size, False),
                        validation_steps=len(x_valid),
                        max_queue_size=1)


def main():
    data_dir = 'data'          # data directory
    test_size = 0.2            # test size fraction
    drop_out = 0.5              # drop out probability
    epochs = 10                # number of epochs
    samples_per_epoch = 20000  # samples per epoch
    batch_size = 40            # batch size
    save_best_only = True      # save best only option
    learning_rate = 1.0e-4     # learning rate

    # print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    print('{:<20} := {}'.format("data_dir:", data_dir))
    print('{:<20} := {}'.format("test_size", test_size))
    print('{:<20} := {}'.format("drop_out", drop_out))
    print('{:<20} := {}'.format("epochs", epochs))
    print('{:<20} := {}'.format("samples_per_epoch", samples_per_epoch))
    print('{:<20} := {}'.format("batch_size", batch_size))
    print('{:<20} := {}'.format("save_best_only", save_best_only))
    print('{:<20} := {}'.format("learning_rate", learning_rate))
    print('-' * 30)

    # load data
    data = loadData(data_dir, test_size)
    # build model
    model = buildModel(drop_out)
    # train model on data, it saves as model.h5 
    trainModel(model, data_dir, batch_size, samples_per_epoch, epochs, save_best_only, learning_rate, *data)

if __name__ == '__main__':
    main()