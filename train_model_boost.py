import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetMobile
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.utils.vis_utils import plot_model

plt.switch_backend('agg')

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

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
        batch_size=64,
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
        batch_size=64,
        seed=2018,
        shuffle=False,
        class_mode='binary',
        target_size=(96, 96)
    )

    return train_generator, valid_generator


def model_bulid():
    inputs = Input(IN_SHAPE)
    conv_base_nasnet = NASNetMobile(
        include_top=False,
        input_shape=IN_SHAPE
    )
    conv_base_resnet = ResNet50(
        include_top=False,
        input_shape=IN_SHAPE
    )
    outputs = Concatenate(axis=-1)([GlobalAveragePooling2D()(conv_base_nasnet(inputs)),
                                    GlobalAveragePooling2D()(conv_base_resnet(inputs))])
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)

    model = Model(inputs, outputs)

    model.compile(Adam(0.01),
                  loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()
    plot_model(model, to_file='NetStruct_boost.png', show_shapes=True)
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
        'weights_boost.h5', monitor='val_loss', save_best_only=True)
    earlystopper = EarlyStopping(
        monitor='val_loss', patience=2, verbose=1)
    reducel = ReduceLROnPlateau(
        monitor='val_loss', patience=1, verbose=1, factor=0.1)

    history = model.fit_generator(train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  epochs=30,
                                  callbacks=[reducel, earlystopper, model_checkpoint, tensorboard])

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('loss_performance_boost.png')
    plt.clf()
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='valid')
    plt.title("model acc")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('acc_performance_boost.png')


if __name__ == '__main__':
    train_model()
