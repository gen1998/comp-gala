from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras import optimizers


class Model_CNN():
    def __init__(self, img_height, img_width, num_classes, model_name, t_learning=False, fine_tuning=False):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.t_learning = t_learning
        self.fine_tuning = fine_tuning
        self.model_name = model_name
        self.model = self.select_model(model_name)

    def select_model(self, model_name):
        if model_name == "vgg16":
            return self.VGG16()
        elif model_name == "mnist_997":
            return self.mnist_997()
        elif model_name == "resnet50":
            return self.resnet50()
        elif model_name == "xception":
            return self.xception()

    def VGG16(self):
        input_shape = (self.img_height, self.img_width, 3)
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
        input_shape = (self.img_height, self.img_width, 3)

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
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        return model

    def resnet50(self):
        input_tensor = Input(shape=(self.img_height, self.img_width, 3))

        if self.fine_tuning or self.t_learning:
            Resnet50 = ResNet50(include_top=False, weights="imagenet" ,input_tensor=input_tensor)
        else:
            Resnet50 = ResNet50(include_top=False, weights=None ,input_tensor=input_tensor)

        if self.t_learning:
            Resnet50.trainable = False

        top_model = Sequential()
        top_model.add(Flatten(input_shape=Resnet50.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(self.num_classes, activation='softmax'))
        top_model = Model(Resnet50.input, top_model(Resnet50.output))

        if self.t_learning:
            top_model.compile(
                loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy']
            )
        else:
            top_model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy']
            )
        return top_model

    def xception(self):
        input_tensor = Input(shape=(self.img_height, self.img_width, 3))

        if self.fine_tuning or self.t_learning:
            xcep = Xception(include_top=False, weights="imagenet" ,input_tensor=input_tensor)
        else:
            xcep = Xception(include_top=False, weights=None ,input_tensor=input_tensor)

        if self.t_learning:
            xcep.trainable = False

        x = xcep.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(xcep.input, predictions)

        if self.t_learning:
            model.compile(
                loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy']
            )
        else:
            model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy']
            )

        return model
