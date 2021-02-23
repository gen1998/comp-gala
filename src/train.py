import numpy as np
import pandas as pd
import os
import pickle
import keras
import cv2
from PIL import Image
import pickle
import shutil
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import LearningRateScheduler


class Model:
    def __init__(self, practie_name, operation, image_size_x=80, image_size_y=80, num_classes=4):
        self.practice_name = practie_name
        self.num_files = {}
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.num_classes = num_classes
        self.batch_size = 0
        self.epoch_size = 0
        self.data_classes = ['Attire', 'misc', 'Food', 'Decorationandsignage']
        #self.model = self.VGG16(operation)
        self.model = self.mnist_997()

    def folder_create(self):
        if os.path.exists("dataset/train/"):
            shutil.rmtree("dataset/train/")

        if os.path.exists("dataset/validation/"):
            shutil.rmtree("dataset/validation/")

        for cla in self.data_classes:
            os.makedirs("dataset/train/"+cla)
            os.makedirs("dataset/validation/"+cla)

    def images_create(self):
        self.folder_create()
        train_csv = pd.read_csv("dataset/train.csv", index_col=0)
        imgs = os.listdir("dataset/Train_Images/")
        for name in imgs:
            img = cv2.imread(os.path.join("dataset/Train_Images/", name))
            img_class = train_csv.loc[name][0]
            img = cv2.resize(img, (self.image_size_x, self.image_size_y))
            if np.random.rand() <= 0.8:
                cv2.imwrite(os.path.join("dataset/train", img_class, name), img)
                cv2.imwrite(os.path.join("dataset/train", img_class, "f_"+name), cv2.flip(img, 1))
            else:
                cv2.imwrite(os.path.join("dataset/validation", img_class, name), img)

    # generator
    def generator(self):
        datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=10,
                                     horizontal_flip=True,
                                     vertical_flip=True)

        train_generator = datagen.flow_from_directory(
            os.path.join("dataset/train"),
            target_size=(self.image_size_x, self.image_size_y),
            batch_size=self.batch_size,
            classes=self.data_classes,
            class_mode="categorical")

        validation_generator = datagen.flow_from_directory(
            os.path.join("dataset/validation"),
            target_size=(self.image_size_x, self.image_size_y),
            batch_size=self.batch_size,
            classes=self.data_classes,
            class_mode="categorical")

        return train_generator, validation_generator

    # categoryごとに学習する
    def train(self,  batch_size=128, epoch_size=100):
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        train_generator, validation_generator = self.generator()
        train_num = 0
        valid_num = 0

        for img_class in self.data_classes:
            train_num += len(os.listdir("dataset/train/"+img_class))
            valid_num += len(os.listdir("dataset/validation/"+img_class))

        print("train_num : ", train_num)
        print("valid_num : ", valid_num)

        """
        cp_cb = ModelCheckpoint(filepath=os.path.join(self.checkpoints_path, category+'checkpoint.h5'),
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='auto'
        """

        """
        def lr_schedul(epoch):
            x = 0.001
            if epoch >= 40:
                x = 0.0001
            if epoch >= 80:
                x = 0.00001
            return x
        """

        # lr_decay = LearningRateScheduler(lr_schedul, verbose=1)
        # es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

        history = self.model.fit_generator(train_generator,
                                           validation_data=validation_generator,
                                           verbose=1,
                                           steps_per_epoch=train_num // self.batch_size,
                                           validation_steps=valid_num // self.batch_size,
                                           epochs=self.epoch_size,
                                           workers=32,
                                           max_queue_size=32)

        self.model.save_weights('src/result/weights/{}_weights.h5'.format(self.practice_name))

        with open("src/result/history/{}_history.pickle".format(self.practice_name), 'wb') as fp:
            pickle.dump(history.history, fp)

    def test(self):
        submission = pd.read_csv("dataset/sample_submission.csv", index_col=0)
        self.model.load_weights('src/result/weights/{}_weights.h5'.format(self.practice_name))
        filenames = list(submission["Image"].values)

        for name in filenames:
            image = Image.open("dataset/test/"+name)
            image = image.convert("RGB")
            image = np.asarray(image, dtype=np.float32)
            image = cv2.resize(image, (self.image_size_x, self.image_size_y))
            image /= 255
            image = np.expand_dims(image, 0)
            result = np.array(self.model.predict(image, batch_size=1, verbose=0)[0])
            submission.loc[submission["Image"] == name, "Class"] = self.data_classes[np.argmax(result)]

        submission.to_csv("src/result/submission/{}_submission.csv".format(self.practice_name), index=False)

    def VGG16(self, operation):
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

        # optimizer
        if operation == "mac":
            optimizer = keras.optimizers.Adam()
        else:
            optimizer = keras.optimizers.adam()

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
        model.add(Dense(10, activation='softmax'))

        # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return model