import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from sklearn.cross_validation import KFold

plt.switch_backend('agg')

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'

train_dir = './dataset/train/'

IMG_SIZE = (96, 96)
IN_SHAPE = (*IMG_SIZE, 3)

dropout_rate = 0.5


def load_data(train, valid):
    train_datagen = ImageDataGenerator(preprocessing_function=lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    valid_datagen = ImageDataGenerator(preprocessing_function=lambda x: (
        x - x.mean()) / x.std() if x.std() > 0 else x)

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


def model_bulid():
    inputs = Input(IN_SHAPE)
    conv_base = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=IN_SHAPE
    )
    x = conv_base(inputs)
    out = Flatten()(x)
    out = Dense(512)(out)
    out = BatchNormalization()(out)
    out = Activation(activation='relu')(out)
    out = Dropout(dropout_rate)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)

    conv_base.Trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'res5a_branch2a':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(Adam(0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()
    plot_model(model, to_file='NetStruct.png', show_shapes=True)
    return model


def train_model():

    nfolds = 5
    epoch = 30
    random_state = 2018

    df_train = pd.read_csv('./dataset/train_labels.csv')
    # df_train = df_train.sample(n=20, random_state=random_state)
    df_train = df_train.values
    kf = KFold(len(df_train), n_folds=nfolds,
               shuffle=True, random_state=random_state)
    num_fold = 0
    for train_index, valid_index in kf:
        train_df, valid_df = df_train[train_index], df_train[valid_index]
        train_df = pd.DataFrame(train_df, columns=['id', 'label'])
        valid_df = pd.DataFrame(valid_df, columns=['id', 'label'])
        train, valid = load_data(train_df, valid_df)
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(train))
        print('Split valid: ', len(valid))
        model = model_bulid()

        STEP_SIZE_TRAIN = train.n//train.batch_size
        STEP_SIZE_VALID = valid.n//valid.batch_size

        tensorboard = TensorBoard(log_dir='./logs_kfold',  # log 目录
                                  # histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                  # batch_size=batch_size,     # 用多大量的数据计算直方图
                                  write_graph=True,  # 是否存储网络结构图
                                  write_grads=False,  # 是否可视化梯度直方图
                                  write_images=False,  # 是否可视化参数
                                  embeddings_freq=0,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)
        model_checkpoint = ModelCheckpoint(
            'weights_kfold'+'_'+str(num_fold)+'.h5', monitor='val_loss', save_best_only=True)
        earlystopper = EarlyStopping(
            monitor='val_loss', patience=2, verbose=1)
        reducel = ReduceLROnPlateau(
            monitor='val_loss', patience=1, verbose=1, factor=0.1)

        history = model.fit_generator(train, steps_per_epoch=STEP_SIZE_TRAIN,
                                      validation_data=valid,
                                      validation_steps=STEP_SIZE_VALID,
                                      epochs=epoch,
                                      callbacks=[reducel, earlystopper, model_checkpoint, tensorboard])

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='valid')
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "valid"], loc="upper left")
        plt.savefig('loss_performance'+'_'+str(num_fold)+'.png')
        plt.clf()
        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='valid')
        plt.title("model acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.legend(["train", "valid"], loc="upper left")
        plt.savefig('acc_performance'+'_'+str(num_fold)+'.png')

        with open('logs_kfold.txt', 'a+') as f:
            f.write(str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))+'\n')
            f.write(str(num_fold)+'\n')
            f.write(str(history.history['loss'])+'\n')
            f.write(str(history.history['val_loss'])+'\n')
            f.write(str(history.history['acc'])+'\n')
            f.write(str(history.history['val_acc'])+'\n')


if __name__ == '__main__':
    train_model()
