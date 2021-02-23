import numpy as np
import pandas as pd
import os
import pickle
import keras
import cv2
from PIL import Image
import pickle
import shutil

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

import src.model as model


class Main(model.Model):
    def __init__(self, practie_name, model_name, image_size_x=80, image_size_y=80, num_classes=4):
        super(Main, self).__init__(image_size_x, image_size_y, num_classes, model_name)
        self.practice_name = practie_name
        self.num_files = {}
        self.num_classes = num_classes
        self.batch_size = 0
        self.epoch_size = 0
        self.data_classes = ['Attire', 'misc', 'Food', 'Decorationandsignage']

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
        s_magnification = 0.6  # 彩度(Saturation)の倍率
        v_magnification = 0.6  # 明度(Value)の倍率

        for name in imgs:
            img = cv2.imread(os.path.join("dataset/Train_Images/", name))
            img_class = train_csv.loc[name][0]
            img = cv2.resize(img, (self.image_size_x, self.image_size_y))
            if np.random.rand() <= 0.8:
                cv2.imwrite(os.path.join("dataset/train", img_class, name), img)
                cv2.imwrite(os.path.join("dataset/train", img_class, "f_"+name), cv2.flip(img, 1))

                # 彩度と明度を変換
                img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*s_magnification  # 彩度の計算
                img_hsv[:,:,(2)] = img_hsv[:,:,(2)]*v_magnification  # 明度の計算
                img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
                cv2.imwrite(os.path.join("dataset/train", img_class, "h_"+name), img_bgr)
                cv2.imwrite(os.path.join("dataset/train", img_class, "h_f_"+name), cv2.flip(img_bgr, 1))
            else:
                cv2.imwrite(os.path.join("dataset/validation", img_class, name), img)

    # generator
    def generator(self):
        datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=10,
                                     zoom_range = 0.10,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
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

        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
        es_cb = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              min_delta=0.01,
                                              patience=20,
                                              verbose=0,
                                              mode='auto',
                                              restore_best_weights=True)

        history = self.model.fit_generator(train_generator,
                                           validation_data=validation_generator,
                                           verbose=1,
                                           steps_per_epoch=train_num // self.batch_size,
                                           validation_steps=valid_num // self.batch_size,
                                           epochs=self.epoch_size,
                                           workers=32,
                                           max_queue_size=32,
                                           callbacks=[annealer, es_cb])

        self.model.save_weights('src/result/weights/{}_weights.h5'.format(self.practice_name))

        with open("src/result/history/{}_history.pickle".format(self.practice_name), 'wb') as fp:
            pickle.dump(history.history, fp)

    def test(self, weights):
        submission = pd.read_csv("dataset/sample_submission.csv", index_col=0)
        filenames = list(submission["Image"].values)
        result = np.zeros((len(filenames), 4))

        for weight in weights:
            self.model.load_weights('src/result/weights/{}_weights.h5'.format(weight))
            for name in filenames:
                image = Image.open("dataset/test/"+name)
                image = image.convert("RGB")
                image = np.asarray(image, dtype=np.float32)
                image = cv2.resize(image, (self.image_size_x, self.image_size_y))
                image /= 255
                image = np.expand_dims(image, 0)
                result += np.array(self.model.predict(image, batch_size=1, verbose=0)[0])

        result /= len(weights)
        submission.loc[submission["Image"] == name, "Class"] = self.data_classes[np.argmax(result)]

        submission.to_csv("src/result/submission/{}_submission.csv".format(self.practice_name), index=False)
