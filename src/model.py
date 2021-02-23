import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization

class Model():
    def __init__(self, image_size_x, image_size_y, num_classes, model_name):
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.num_classes = num_classes
        if model_name == "vgg16":
            self.model = self.VGG16()
        elif model_name == "mnist_997":
            self.model = self.mnist_997()

    def VGG16(self):
        input_shape = (self.image_size_x, self.image_size_y, 3)
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=input_shape, name='block1_conv1'))
        model.add(BatchNormalization(name='bn1'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='block1_conv2'))
        model.add(BatchNormalization(name='bn2'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block1_pool'))
        #model.add(Dropout(0.5))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', name='block2_conv1'))
        model.add(BatchNormalization(name='bn3'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', name='block2_conv2'))
        model.add(BatchNormalization(name='bn4'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block2_pool'))
        #model.add(Dropout(0.5))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='block3_conv1'))
        model.add(BatchNormalization(name='bn5'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='block3_conv2'))
        model.add(BatchNormalization(name='bn6'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='block3_conv3'))
        model.add(BatchNormalization(name='bn7'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block3_pool'))
        #model.add(Dropout(0.5))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block4_conv1'))
        model.add(BatchNormalization(name='bn8'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block4_conv2'))
        model.add(BatchNormalization(name='bn9'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block4_conv3'))
        model.add(BatchNormalization(name='bn10'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block4_pool'))
        #model.add(Dropout(0.5))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block5_conv1'))
        model.add(BatchNormalization(name='bn11'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block5_conv2'))
        model.add(BatchNormalization(name='bn12'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block5_conv3'))
        model.add(BatchNormalization(name='bn13'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block5_pool'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(units=4096, activation='relu', name='fc1'))
        model.add(Dense(units=4096, activation='relu', name='fc2'))
        model.add(Dense(units=self.num_classes, activation='softmax', name='predictions'))
        model.summary()

        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def mnist_997(self):
        model = Sequential()
        input_shape = (self.image_size_x, self.image_size_y, 3)

        model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size = 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(64, kernel_size = 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size = 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, kernel_size = 4, activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))

        # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return model