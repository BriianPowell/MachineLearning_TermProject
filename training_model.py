import numpy as np 

#data analysis library used to read rows and columns in dataset
import pandas as pd 

#splits arrays and matrices into random train and test subsets
from sklearn.model_selection import train_test_split 

#neural network library
import keras
#linear stack of layers
from keras.models import Sequential 

#optimization for gradient descent
from keras.optimizers import Adam 

#save model with ModelCheckPoint
from keras.callbacks import ModelCheckpoint

#CNN Layers
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout 

#helper methods
from helper import INPUT_SHAPE, batch_generator 

#parser for command line options, arguments and sub-commands
import argparse 

#read files
import os 

np.random.seed(0)


def loadData(args):

	
    #reads CSV dataset into a dataframe = data_df 
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
	
    #store camera images into input data, x
    #images include center, right, and left
    x = data_df[['center', 'left', 'right']].values
	
    #store the steering angle data as output, y
    y = data_df['steering'].values
	
    #Splits training and testing sets 
    x_train, x_valid,  y_train, y_valid = train_test_split(x, y, test_size=args.test_size, random_state=0)

    return x_train, x_valid, y_train, y_valid

def buildModel(args):

    '''
    NVIDIA's CNN Architecture
	
    First layer performs image normalization followed by 5 convolutional layers.
	
    Dropout is added to avoid overfitting.
	
    Finally, 3 fully connected layers leads to a final output steering command.
	
    The fully connected layers are designed to function as a controller for steering.
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE)) #normalization
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def trainModel(model, args, x_train, x_valid, y_train, y_valid):

	
    #ModelCheckPoint saves each model after every epoch
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto') #model's weights will be saved
	
    #mean squared error used to calculate loss
    #Adam optimization algorithm for stochastic gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    #trains model on data generated batch by batch
    model.fit_generator(batch_generator(args.data_dir, x_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.epochs,
                        verbose=1,
                        callbacks=[checkpoint],
                        validation_data=batch_generator(args.data_dir, x_valid, y_valid, args.batch_size, False),
                        validation_steps=len(x_valid),
                        max_queue_size=1)

#helper method converts string to boolean
def s2B(s):
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

def main():
    #Command line interface for training/validation dataset
    parser = argparse.ArgumentParser(description='Self Driving Car Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='epochs',            type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2B,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = loadData(args)
    #build model
    model = buildModel(args)
    #train model on data, it saves as model.h5 
    trainModel(model, args, *data)

if __name__ == '__main__':
    main()

"""
OUTPUT:
Using TensorFlow backend.
------------------------------
Parameters
------------------------------
data_dir             := data
test_size            := 0.2
keep_prob            := 0.5
epochs               := 10
samples_per_epoch    := 20000
batch_size           := 40
save_best_only       := True
learning_rate        := 0.0001
------------------------------
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 66, 200, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 18, 64)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
2018-12-02 16:53:13.717267: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2018-12-02 16:53:14.047689: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1405] Found device 0 with properties:
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.7845
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.63GiB
2018-12-02 16:53:14.060618: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1484] Adding visible gpu devices: 0
2018-12-02 16:53:16.236745: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-12-02 16:53:16.245067: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:971]      0
2018-12-02 16:53:16.247641: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:984] 0:   N
2018-12-02 16:53:16.252734: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6391 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1)
20000/20000 [==============================] - 1889s 94ms/step - loss: 0.0262 - val_loss: 0.0175
Epoch 2/10
20000/20000 [==============================] - 1924s 96ms/step - loss: 0.0216 - val_loss: 0.0164
Epoch 3/10
20000/20000 [==============================] - 1966s 98ms/step - loss: 0.0203 - val_loss: 0.0175
Epoch 4/10
20000/20000 [==============================] - 1851s 93ms/step - loss: 0.0195 - val_loss: 0.0166
Epoch 5/10
20000/20000 [==============================] - 1826s 91ms/step - loss: 0.0189 - val_loss: 0.0165
Epoch 6/10
20000/20000 [==============================] - 1914s 96ms/step - loss: 0.0187 - val_loss: 0.0165
Epoch 7/10
20000/20000 [==============================] - 1945s 97ms/step - loss: 0.0182 - val_loss: 0.0169
Epoch 8/10
20000/20000 [==============================] - 1906s 95ms/step - loss: 0.0179 - val_loss: 0.0171
Epoch 9/10
20000/20000 [==============================] - 1901s 95ms/step - loss: 0.0176 - val_loss: 0.0166
Epoch 10/10
20000/20000 [==============================] - 1888s 94ms/step - loss: 0.0173 - val_loss: 0.0165
"""