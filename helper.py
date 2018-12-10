import cv2, os
import numpy as np
import matplotlib.image as mpimg


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    '''
    Load RGB images from a file
    '''
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
    '''
    Crop the image to remove the sky at the top and the front bumper of the car at the bottom
    '''
    return image[60:-25, :, :]


def resize(image):
    '''
    Resize the image to the input shape used by the network model
    '''
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    '''
    Convert the image from RGB to YUV (NVidia Model does the same)
    '''
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    '''
    Atomic preprocess function
    '''
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    '''
    Randomly choose an image from the center, left or right, and adjust the steering angle accordingly
    '''
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    '''
    Generate training image give image paths and associated steering angles
    '''
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # Randomly selecting an image to train on
            if is_training: #and np.random.rand() < 0.6:
                image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
            else:
                # If not training just load the image
                image = load_image(data_dir, center) 
            # Add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers