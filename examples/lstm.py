# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import optimizers, Sequential, Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from numpy.random import seed
from matplotlib.pylab import rcParams
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import set_random_seed
from keras import backend as K

# Import DeepExplain
from deepexplain.tensorflow import DeepExplain

seed(7)
set_random_seed(11)
# tf.random.set_seed(11)

# used to help randomly select the data points
SEED = 123
DATA_SPLIT_PCT = 0.2

rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal", "Break"]

'''
Download data here:
https://docs.google.com/forms/d/e/1FAIpQLSdyUk3lfDl7I5KYK_pw285LCApc-_RcoC0Tf9cnDnZ_TWzPAw/viewform
'''
df = pd.read_csv("data/processminer-rare-event-mts - data.csv")
df.head(n=5)  # visualize the data.

sign = lambda x: (1, -1)[x < 0]


def curve_shift(df, shift_by):
    """
    将异常标记上移/下移shift行，并且去除原始异常标记行，即删除异常进行时只标记异常发生前/后的几秒作为异常
    This function will shift the binary labels in a dataframe.
    The curve shift will be with respect to the 1s.
    For example, if shift is -2, the following process
    will happen: if row n is labeled as 1, then
    - Make row (n+shift_by):(n+shift_by-1) = 1.
    - Remove row n.
    i.e. the labels will be shifted up to 2 rows up.

    Inputs:
    df       A pandas dataframe with a binary labeled column.
             This labeled column should be named as 'y'.
    shift_by An integer denoting the number of rows to shift.

    Output
    df       A dataframe with the binary labels shifted by shift.

    这个函数是用来偏移数据中的二分类标签。
    平移只针对标签为 1 的数据
    举个例子，如果偏移量为 -2，下面的处理将会发生：
    如果是 n 行的标签为 1，那么
    - 使 (n+shift_by):(n+shift_by-1) = 1
    - 删除第 n 行。
    也就是说标签会上移 2 行。

    输入：
    df       一个分类标签列的 pandas 数据。
             这个标签列的名字是 ‘y’。
    shift_by 一个整数，表示要移动的行数。

    输出：
    df       按照偏移量平移过后的数据。
    """

    vector = df['y'].copy()
    for s in range(abs(shift_by)):
        tmp = vector.shift(sign(shift_by))  # shift函数是对数据进行移动的操作
        tmp = tmp.fillna(0)  # Nah的地方填充0
        vector += tmp
    labelcol = 'y'
    # Add vector to the df
    df.insert(loc=0, column=labelcol + 'tmp', value=vector)
    # Remove the rows with labelcol == 1.去掉原始异常项
    df = df.drop(df[df[labelcol] == 1].index)
    # Drop labelcol and rename the tmp col as labelcol
    df = df.drop(labelcol, axis=1)
    df = df.rename(columns={labelcol + 'tmp': labelcol})
    # Make the labelcol binary
    df.loc[df[labelcol] > 0, labelcol] = 1

    return df


'''
Shift the data by 2 units, equal to 4 minutes.

Test: Testing whether the shift happened correctly.
'''
print('Before shifting')  # Positive labeled rows before shifting.
one_indexes = df.index[df['y'] == 1]
print(df.iloc[(one_indexes[0] - 3):(one_indexes[0] + 2), 0:5].head(n=5))

# Shift the response column y by 2 rows to do a 4-min ahead prediction.
# df = curve_shift(df, shift_by = -2)


print('After shifting')  # Validating if the shift happened correctly.
print(df.iloc[(one_indexes[0] - 4):(one_indexes[0] + 1), 0:5].head(n=5))

# Remove time column, and the categorical columns
df = df.drop(['time', 'x28', 'x61'], axis=1)

'''
Prepare data for LSTM models
LSTM is a bit more demanding than other models. Significant amount of time 
and attention goes in preparing the data that fits an LSTM.

First, we will create the 3-dimensional arrays of shape: 
(samples x timesteps x features). Samples mean the number of data points. 
Timesteps is the number of time steps we look back at any time t to make a
 prediction. This is also referred to as lookback period. The features is the
  number of features the data has, in other words, the number of predictors in 
  a multivariate data.

'''
input_X = df.loc[:, df.columns != 'y'].values  # converts the df to a numpy array
input_y = df['y'].values
print("input_X.shape:")
print(input_X.shape)
n_features = input_X.shape[1]  # number of features
print("n_features:")
print(n_features)


# 切成时间片长为5，步长为1的时间片的集合
def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    print(len(X))
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # Gather past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
            # print(np.shape(X[[(i+j+1)], :]))
        output_X.append(t)
        output_y.append(y[i + lookback + 1])
    return output_X, output_y


'''
In LSTM, to make prediction at any time t, we will look at data from (t-lookback):t. In the following, we have an example to show how the input data are transformed with the `temporalize` function with `lookback=5`. For the modeling, we may use a longer lookback.
'''
'''
Test: The 3D tensors (arrays) for LSTM are forming correctly.
'''
print('First instance of y = 1 in the original data')
print(df.iloc[(np.where(np.array(input_y) == 1)[0][0] - 5):(np.where(np.array(input_y) == 1)[0][0] + 1), ])

lookback = 5  # Equivalent to 10 min of past data.
# Temporalize the data
X, y = temporalize(X=input_X, y=input_y, lookback=lookback)
print("X.size")
print(np.shape(X))
print("y.size")
print(np.shape(y))

print('For the same instance of y = 1, we are keeping past 5 samples in the 3D predictor array, X.')
print(pd.DataFrame(np.concatenate(X[np.where(np.array(y) == 1)[0][0]], axis=0)))  # 输出shape：（5，59）,concatenate纵向拼接
print(pd.DataFrame(np.concatenate(X[np.where(np.array(y) == 1)[0][0]], axis=0)).shape)

# The two tables are the same. This testifies that we are correctly taking 5 samples (= lookback), X(t):X(t-5) to
# predict y(t). ## Divide the data into train, valid, and test
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=DATA_SPLIT_PCT,
                                                    random_state=SEED)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=DATA_SPLIT_PCT, random_state=SEED)
print("X_train:")
print(X_train.shape)
print("X_valid:")
print(X_valid.shape)
print("X_test:")
print(X_test.shape)
print(type(X_train))

X_train_y0 = X_train[y_train == 0]
X_train_y1 = X_train[y_train == 1]

X_valid_y0 = X_valid[y_valid == 0]
X_valid_y1 = X_valid[y_valid == 1]
print("X_train_y0")
print(X_train_y0.shape)
print("X_train_y1")
print(X_train_y1.shape)
print("X_valid_y0")
print(X_valid_y0.shape)
print("X_valid_y1")
print(X_valid_y1.shape)

# # Reshaping the data The tensors we have here are 4-dimensional. We will reshape them into the desired 3-dimensions
# corresponding to sample x lookback x features.
# #重塑数据 这里的张量是4维的。
# 我们将它们重塑为所需的3维，对应于样本x回溯x特征。
X_train = X_train.reshape(X_train.shape[0], lookback, n_features)
X_train_y0 = X_train_y0.reshape(X_train_y0.shape[0], lookback, n_features)
X_train_y1 = X_train_y1.reshape(X_train_y1.shape[0], lookback, n_features)

X_test = X_test.reshape(X_test.shape[0], lookback, n_features)

X_valid = X_valid.reshape(X_valid.shape[0], lookback, n_features)
X_valid_y0 = X_valid_y0.reshape(X_valid_y0.shape[0], lookback, n_features)
X_valid_y1 = X_valid_y1.reshape(X_valid_y1.shape[0], lookback, n_features)

'''
### Standardize the data
It is usually better to use a standardized data (transformed to Gaussian, mean 0 and sd 1) for autoencoders.

One common mistake is: we normalize the entire data and then split into train-test. This is not correct. Test data should be completely unseen to anything during the modeling. We should normalize the test data using the feature summary statistics computed from the training data. For normalization, these statistics are the mean and variance for each feature. 

The same logic should be used for the validation set. This makes the model more stable for a test data.

To do this, we will require two UDFs.

- `flatten`: This function will re-create the original 2D array from which the 3D arrays were created. This function is the inverse of `temporalize`, meaning `X = flatten(temporalize(X))`.
- `scale`: This function will scale a 3D array that we created as inputs to the LSTM.
'''


def flatten(X):
    """
    Flatten a 3D array.

    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.

    Output
    flattened_X  A 2D array, sample x features.
    """
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
        # print((X.shape[1] - 1))
    # print(flattened_X.shape)
    return (flattened_X)


def scale(X, scaler):
    """
    Scale 3D array.

    Inputs
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize

    Output
    X            Scaled 3D array.

    preprocessing这个模块还提供了一个实用类StandarScaler，它可以在训练数据集上做了标准转换操作之后，
    把相同的转换应用到测试训练集中。
    可以对训练数据，测试数据应用相同的转换，以后有新的数据进来也可以直接调用，不用再重新把数据放在一起再计算一次了。
    """
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])

    return X


# Initialize a scaler using the training data.
scaler = StandardScaler().fit(flatten(X_train_y0))  # 用展平的X_train_y0训练StandardScaler转换器

X_train_y0_scaled = scale(X_train_y0, scaler)  # 标准化X_train_y0
X_train_y1_scaled = scale(X_train_y1, scaler)  # 标准化X_train_y1
X_train_scaled = scale(X_train, scaler)  # 标准化X_train

'''
Test: Check if the scaling is correct.

The test succeeds if all the column means 
and variances are 0 and 1, respectively, after
flattening.
测试：检查缩放比例是否正确。

展平后，如果所有列均值和方差分别为0和1，则测试成功。
------此处有一个逻辑问题，也许应该用X_train_scaled来测试
'''
a = flatten(X_train_y0_scaled)  # 展平
print('colwise mean', np.mean(a, axis=0).round(6))  # 计算展平后均值
print('colwise variance', np.var(a, axis=0))  # 计算展平后方差

# The test succeeded. Now we will _scale_ the validation and test sets.
# 测试成功。 现在我们将缩放验证和测试集。
X_valid_scaled = scale(X_valid, scaler)
X_valid_y0_scaled = scale(X_valid_y0, scaler)

X_test_scaled = scale(X_test, scaler)

## LSTM Autoencoder training
# First we will initialize the Autoencoder architecture. We are building a simple autoencoder. More complex architectures and other configurations should be explored.
timesteps = X_train_y0_scaled.shape[1]  # equal to the lookback
n_features = X_train_y0_scaled.shape[2]  # 59

epochs = 200
batch = 64
lr = 0.0001

lstm_autoencoder = Sequential()
# Encoder
lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
lstm_autoencoder.add(RepeatVector(timesteps))
# lstm_autoencoder.add()
# Decoder
lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

lstm_autoencoder.summary()

'''
As a rule-of-thumb, look at the number of parameters. If not using any regularization, keep this less than the 
number of samples. If using regularization, depending on the degree of regularization you can let more parameters in 
the model that is greater than the sample size. For example, if using dropout with 0.5, you can have up to double the 
sample size (loosely speaking). 
'''
adam = optimizers.Adam(lr)
lstm_autoencoder.compile(loss='mse', optimizer=adam)

cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                     save_best_only=True,
                     verbose=0)

tb = TensorBoard(log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=True)

lstm_autoencoder_history = lstm_autoencoder.fit(X_train_y0_scaled, X_train_y0_scaled,
                                                epochs=epochs,
                                                batch_size=batch,
                                                validation_data=(X_valid_y0_scaled, X_valid_y0_scaled),
                                                verbose=2).history

# plt show
plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
plt.plot(lstm_autoencoder_history['val_loss'], linewidth=2, label='Valid')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

train_x_predictions = lstm_autoencoder.predict(X_train_scaled)
mse = np.mean(np.power(flatten(X_train_scaled) - flatten(train_x_predictions), 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': y_train.tolist()})

groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label="Break" if name == 1 else "Normal")
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

valid_x_predictions = lstm_autoencoder.predict(X_valid_scaled)
mse = np.mean(np.power(flatten(X_valid_scaled) - flatten(valid_x_predictions), 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': y_valid.tolist()})

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

test_x_predictions = lstm_autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(flatten(X_test_scaled) - flatten(test_x_predictions), 2), axis=1)  # 计算平均方差

error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': y_test.tolist()})

threshold_fixed = 0.3
groups = error_df.groupby('True_class')  # 根据True_class即实际的0，1分类
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label="Break" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.True_class, pred_y)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate, )
print("false_pos_rate")
print(false_pos_rate)
print("true_pos_rate")
print(true_pos_rate)
print("thresholds")
print(thresholds)
print("roc_auc")
print(roc_auc)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


