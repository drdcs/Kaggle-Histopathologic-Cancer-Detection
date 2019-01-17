from keras.models import load_model
from keras import backend as K
import os
from glob import glob
import numpy as np
import pandas as pd
from skimage.io import imread

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

if __name__ == '__main__':

    base_test_dir = './dataset/test/'
    model = load_model('weights.h5')
    # model = load_model('weights_nasnet.h5')
    # model = load_model('weights_boost.h5')
    print(model.summary())

    test_files = glob(os.path.join(base_test_dir, '*.tif'))
    submission = pd.DataFrame()
    file_batch = 5000
    max_idx = len(test_files)
    for idx in range(0, max_idx, file_batch):
        print("Indexes: %i - %i" % (idx, idx+file_batch))
        test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})
        test_df['id'] = test_df.path.map(
            lambda x: x.split('/')[3].split(".")[0])
        test_df['image'] = test_df['path'].map(imread)
        K_test = np.stack(test_df["image"].values)
        K_test = (K_test - K_test.mean()) / K_test.std()
        predictions = model.predict(K_test)
        test_df['label'] = predictions
        submission = pd.concat([submission, test_df[["id", "label"]]])
    print(submission.head())
    submission.to_csv("submission.csv", index=False, header=True)
