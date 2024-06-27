# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
# import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, plot_model
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report
# from sklearn import metrics
# from sklearn.metrics import roc_curve, auc, precision_recall_curve
# from scipy.interpolate import interp1d

df = pd.read_csv(r"data_1.csv")
df2 = pd.read_csv(r"data_2.csv")
X = np.expand_dims(df.values[:, 0:450].astype(float), axis=2)
Y = df.values[:, 450]
X_e = np.expand_dims(df2.values[:, 0:450].astype(float), axis=2)
Y_e = df2.values[:, 450]
# X_data = df2.iloc[:,:-1].values
# Y_data = df2.iloc[:,-1].values.astype('uint8')

encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)
Y_e_encoded = encoder.fit_transform(Y_e)
Y_e_onehot = np_utils.to_categorical(Y_e_encoded)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=0, stratify=Y_onehot)

def baseline_model():
    # model = Sequential()
    # model.add(Conv1D(3, 1, input_shape=(450, 1)))
    # model.add(Conv1D(3, 1, activation='tanh'))
    # model.add(MaxPooling1D(1))
    # model.add(Conv1D(4, 1, activation='tanh'))
    # model.add(Conv1D(4, 1, activation='tanh'))
    # model.add(MaxPooling1D(1))
    # model.add(Conv1D(6, 1, activation='tanh'))
    # model.add(Conv1D(6, 1, activation='tanh'))
    # model.add(MaxPooling1D(1))
    # model.add(Flatten())
    model = Sequential()
    model.add(Conv1D(3, 1, input_shape=(450, 1)))
    model.add(Conv1D(3, 1, activation='tanh'))
    model.add(MaxPooling1D(1))
    # model.add(Conv1D(12, 1, activation='tanh'))
    # model.add(Conv1D(12, 1, activation='tanh'))
    # model.add(MaxPooling1D(1))
    model.add(Conv1D(4, 1, activation='tanh'))
    model.add(Conv1D(4, 1, activation='tanh'))
    model.add(MaxPooling1D(1))
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))
    # plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=800, batch_size=80, verbose=1)
estimator.fit(X_train, Y_train)

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.rcParams['font.weight'] = 'bold'
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = '12'
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, ('0%', '3%', '5%', '8%', '10%', '12%', '15%', '18%', '20%', '25%'))
    # plt.yticks(tick_marks, ('0%', '3%', '5%', '8%', '10%', '12%', '15%', '18%', '20%', '25%'))
    plt.xticks(tick_marks, ('1', '2', '3', '4'))
    plt.yticks(tick_marks, ('1', '2', '3', '4'))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 300 # plt.show()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, hspace=0.2, wspace=0.3)
    plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()

def plot_confuse(model, x_val, y_val):
    predicted = loaded_model.predict(x_val)
    predictions = np.argmax(predicted, axis=1)
    truelabel = y_val.argmax(axis=-1)
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))

# json
model_json = estimator.model.to_json()
with open(r"D:\Projects\yolov5-7.0-IOU\model.json",'w')as json_file:
    json_file.write(model_json)
estimator.model.save_weights('model.h5')

json_file = open(r"D:\Projects\yolov5-7.0-IOU\model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
# predicted = loaded_model.predict(X)
# predicted_label = loaded_model.predict_classes(X)
predicted = loaded_model.predict(X)
predicted_label = np.argmax(predicted,axis=1)
# print("predicted label:\n " + str(predicted_label))
predicted_test = loaded_model.predict(X_test)
predicted_label_test = np.argmax(predicted_test,axis=1)
# print("predicted label:\n ", predicted_label_test)
y_test = Y_test.argmax(axis=-1)
# print("Y_test:\n " , y_test)
report = classification_report(y_test, predicted_label_test)
print("report", report)

# PLOT CONFUSE
plot_confuse(estimator.model, X_test, Y_test)

# classes = ['1', '2', '3', '4']
# confusion_matrix = np.array(
#     [[14,  0, 0,  0],
#      [0,  14, 0,  0],
#      [0,  0, 13,  1],
#      [1,  0, 0,  13],
#      ], dtype=np.int64)
# proportion = []
# length = len(confusion_matrix)
# print(length)
# for i in confusion_matrix:
#     for j in i:
#         temp = j / (np.sum(i))
#         proportion.append(temp)
# # print(np.sum(confusion_matrix[0]))
# # print(proportion)
# pshow = []
# for i in proportion:
#     pt = "%.2f%%" % (i * 100)
#     pshow.append(pt)
# proportion = np.array(proportion).reshape(length, length)
# pshow = np.array(pshow).reshape(length, length)
# # print(pshow)
# config = {
#     "font.family": 'Arial',
#     # "font.size": 15,
# }
# rcParams.update(config)
# plt.rcParams['font.weight'] = 'bold'
# plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)
# # ('Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
# # plt.title('confusion_matrix')
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, fontsize=12)
# plt.yticks(tick_marks, classes, fontsize=12)
#
# iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
# for i, j in iters:
#     if (i == j):
#         plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
#                  weight=5)
#         plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
#     else:
#         plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)
#         plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10)
# plt.rcParams['savefig.dpi'] = 600
# plt.rcParams['figure.dpi'] = 300
# # plt.ylabel('True label', fontsize=16)
# # plt.xlabel('Predict label', fontsize=16)
# plt.tight_layout()
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,hspace=0.2, wspace=0.3)
# # plt.savefig('fig.png')
# plt.savefig('figconfuse.png')
# plt.show()

predicted_shiyan = loaded_model.predict(X_e)
predicted_label_shiyan = np.argmax(predicted_shiyan,axis=1)
# print("predicted label shiyan:\n ", predicted_label_shiyan)
y_e = Y_e_onehot.argmax(axis=-1)
# print("y_e_shiyan:\n " , y_e)
# print("Y_data_shiyan:\n " , Y_data)

if predicted_label_shiyan[0] == y_e[0]:
    if predicted_label_shiyan[0] == 2:
            A = X_e[0,100:250]
            # A = X_e[0, 150:200]
            B = X_e[0,350:450]
            a = A.max()
            b = B.max()
            Pb_1 = (a*1000+0.131)/0.862
            Hg_1 = (b*1000- 0.978)/0.387
            print("Detection result of Pb:", Pb_1)
            print("Detection result of Hg:", Hg_1)
            # print("b", b)
    elif predicted_label_shiyan[0] == 0:
            C = X_e[0, 100:250]
            c = C.max()
            Pb_2 = (c*1000+0.131)/0.862
            print("Detection result of Pb:", Pb_2)
    elif predicted_label_shiyan[0] == 1:
            D = X_e[0, 350:450]
            d = D.max()
            Hg_2 = (d*1000- 0.978)/0.387
            print("Detection result of Hg:", Hg_2)
    else:
            print("UNABLE TO DETECT")
else:
    print("UNABLE TO DETECT")