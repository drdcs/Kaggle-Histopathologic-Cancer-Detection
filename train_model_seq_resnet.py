'''
@Description:
@Author: HuangQinJian
@Date: 2018-12-16 15:57:44
@LastEditTime: 2019-01-19 20:38:25
@LastEditors: HuangQinJian
'''
import os
import tensorflow as tf
from keras import backend as K
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.activations import relu, sigmoid
from keras.layers import multiply, Dense, Flatten, add, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Input, ZeroPadding2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.utils.vis_utils import plot_model

plt.switch_backend('agg')

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5'

train_dir = './dataset/train/'
# test_dir = './dataset/test/'

IMG_SIZE = (96, 96)
IN_SHAPE = (*IMG_SIZE, 3)

dropout_rate = 0.5


def load_data():
    df_train = pd.read_csv('./dataset/train_labels.csv')

    # df = df_train.sample(n=100, random_state=2018)
    df = df_train  # using full dataset
    train, valid = train_test_split(df, test_size=0.2)

    train_datagen = ImageDataGenerator(preprocessing_function=lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    valid_datagen = ImageDataGenerator(preprocessing_function=lambda x: (
        x - x.mean()) / x.std() if x.std() > 0 else x)

    # train_datagen = ImageDataGenerator()

    # valid_datagen = ImageDataGenerator()

    # use flow_from_dataframe method to build train and valid generator
    # Only shuffle the train generator as we want valid generator to have the same structure as test

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=train_dir,
        x_col='id',
        y_col='label',
        has_ext=False,
        # subset='training',
        batch_size=32,
        seed=2018,
        shuffle=True,
        class_mode='binary',
        target_size=(96, 96))

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid,
        directory=train_dir,
        x_col='id',
        y_col='label',
        has_ext=False,
        # subset='validation',
        batch_size=32,
        seed=2018,
        shuffle=False,
        class_mode='binary',
        target_size=(96, 96)
    )

    return train_generator, valid_generator


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding,
               strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def bottleneck_Block(inpt, nb_filters, with_seq=False, with_shortcut=False, strides=(1, 1), inpt_c=0):
    k1, k2, k3 = nb_filters
    factor_k = int(inpt_c/16)
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1,
                  strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_seq and with_shortcut:

        shortcut = GlobalAveragePooling2D()(x)

        shortcut = Dense(factor_k, activation='relu')(shortcut)

        shortcut = Dense(k3, activation='sigmoid')(shortcut)

        shortcut = multiply([shortcut, x])

        inpt = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)

        x = add([inpt, shortcut])
    else:
        x = add([x, inpt])
    return x


def resnet_50(classes):
    inpt = Input(IN_SHAPE)
    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(
        7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # conv2_x
    x = bottleneck_Block(x, nb_filters=[64, 64, 256], strides=(
        1, 1), inpt_c=64, with_seq=True, with_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64, 64, 256], inpt_c=256)
    x = bottleneck_Block(x, nb_filters=[64, 64, 256], inpt_c=256)

    # conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512], strides=(
        2, 2), inpt_c=256, with_seq=True, with_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512], inpt_c=512)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512], inpt_c=512)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512], inpt_c=512)

    # conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024], strides=(
        2, 2),  inpt_c=512, with_seq=True, with_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024], inpt_c=1024)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024], inpt_c=1024)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024], inpt_c=1024)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024], inpt_c=1024)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024], inpt_c=1024)

    # conv5_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=(
        2, 2), inpt_c=1024, with_seq=True, with_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], inpt_c=2048)
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], inpt_c=2048)

    x = MaxPooling2D(pool_size=2)(x)
    # x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model


def model_bulid():
    model = resnet_50(classes=2)
    model.compile(Adam(0.01),
                  loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()
    plot_model(model, to_file='NetStruct.png', show_shapes=True)
    return model


def train_model():

    model = model_bulid()
    train_generator, valid_generator = load_data()

    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

    tensorboard = TensorBoard(log_dir='./logs',  # log 目录
                              # histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                              # batch_size=batch_size,     # 用多大量的数据计算直方图
                              write_graph=True,  # 是否存储网络结构图
                              write_grads=False,  # 是否可视化梯度直方图
                              write_images=False,  # 是否可视化参数
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)
    model_checkpoint = ModelCheckpoint(
        'weights.h5', monitor='val_loss', save_best_only=True)
    earlystopper = EarlyStopping(
        monitor='val_loss', patience=2, verbose=1)
    reducel = ReduceLROnPlateau(
        monitor='val_loss', patience=1, verbose=1, factor=0.1)

    history = model.fit_generator(train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  epochs=15,
                                  callbacks=[reducel, earlystopper, model_checkpoint, tensorboard])

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('loss_performance.png')
    plt.clf()
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='valid')
    plt.title("model acc")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('acc_performance.png')


if __name__ == '__main__':
    train_model()
