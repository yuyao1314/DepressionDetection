import glob,os
import cv2
import math
import numpy as np
import tensorflow as tf
import csv
import pandas as pd
import keras
from tqdm import tqdm
from keras_preprocessing import image
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc
from keras.models import *
from keras.layers import *
from keras.layers import add
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras import metrics,losses
from collections import Counter
from itertools import cycle




# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#图片数据路径         /media/som/HFUT/slice
data = ['223_1',
'223_2',
'225_2',
'226_1',
'227_2',
'228_1',
'229_2',
'230_1',
'232_1',
'233_1',
'234_3',
'236_1',
'237_3',
'238_2',
'239_1',
'240_1',
'240_2',
'241_2',
'242_3',
'243_1',
'306_3',
'308_3',
'310_4',
'312_2',
'317_1',
'317_3',
'318_2',
'318_3',
'320_1',
'320_2',
'321_2',
'322_1',
'324_1',
'329_1',
'331_1',
'332_2',
'332_4']
# data = ['7-2',
#         '7-3',
#         '7-4', '7-5',
#         '7-6', '7-7',
#         '7-8', '7-9', '7-10', '7-11', '7-12', '7-13', '7-14',
#         '7-15', '7-16', '7-17', '7-18', '7-19', '7-20', '7-21',
#         '7-22', '7-23', '7-24', '7-25', '7-26', '7-27', '7-28',
#         '7-29', '7-30', '7-31', '8-1', '8-2', '8-3', '8-4',
#         '8-5', '8-6', '8-7', '8-8', '8-9', '8-10', '8-11',
#         '8-12', '8-13', '8-14', '8-15', '8-16', '8-17', '8-18',
#         '8-19', '8-20', '8-21', '8-22', '8-23', '8-24', '8-25',
#         '8-26', '8-27', '8-28', '8-29', '8-30', '8-31', '9-1',
#         '9-2', '9-3', '9-4', '9-5', '9-6', '9-7', '9-8',
#         '9-9', '9-10', '9-11', '9-12', '9-13', '9-14', '9-15',
#         '9-16', '9-17', '9-18', '9-19', '9-20', '9-21', '9-22',
#         '9-23', '9-24', '9-25', '9-26', '9-27', '9-28', '9-29',
#         '9-30', '10-1', '10-2', '10-3', '10-4', '10-5', '10-6',
#         '10-7', '10-8', '10-9', '10-10', '10-11', '10-12', '10-13',
#         '10-14', '10-15', '10-16', '10-17', '10-18', '10-19', '10-20',
#         '10-21', '10-22', '10-23', '10-24', '10-25', '10-26', '10-27',
#         '10-28', '10-29', '10-30', '10-31', '11-1', '11-2', '11-3',
#         '11-4', '11-5', '11-6', '11-7', '11-8', '11-9', '11-10',
#         '11-11', '11-12', '11-13', '11-14', '11-15']
train_set = []
#thermal_train_set = []
score=[]
poms_set=[]
volun = []
n = 0
for i in tqdm(range(len(data))):
    original1 = r'/home/som/lab-data/yuyaodata/2013test'   #可见光图像路径
   # original3 = r'/home/som/lab-data/thermal_img/'     #热红外图像路径
    original1_path = original1+'/'+data[i]
    #original3_path = original3+'/'+data[i]             #热红外图像
    original2 = r'/home/som/lab-data/yuyaodata/Training_Depression_Labels'
    original2_path = original2 + '/' + data[i] + '.csv'
    volunteer=os.listdir(original1_path)
    print("停止日期：",data[i])
    #for x in range(len(volunteer)):
        #n +=1
        #df = pd.read_csv(original2_path, encoding='gbk')
        #df['name'] = data[i]
        #score.append(df)
        #df=pd.concat(score, axis=0)
        #temp = df[(df['编号'] == volunteer[x]) & (df['name'] == data[i])]
        #temp1 = temp.values[0][1:-2]
        #poms_set.append(temp1)     #POMS分量
        # print("POMSSSSSSSSS",poms_set)
        each_data1=os.listdir(original1_path)
        #each_data3=os.listdir(original3_path)
        # print(each_data3)
        # print("停止位置：",volunteer[x])
        #k = 0        len(each_data)//400)
        # print("XXXXX",len(each_data))
        # for y in range(0,len(each_data),30):
####可见光图像trainset
        for y in range(61 , 361):
            #if k >= 2000:
                #breakeach_data1[y]
            detail=original1_path+'/'+volunteer[x]+'/'+'{0}'.format(y)+'.jpg'
            # print("可见光路径：",detail)
            #k+=1
            img1 = cv2.imread(detail)
            # res1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            res1 = cv2.resize(img1,(128,128))
            train_set.append(res1)
###热红外图像trainset         len(each_data3)
        #for t in range(51 ,301 ):
            #if k >= 2000:
                #break
            #detai3=original3_path+'/'+volunteer[x]+'/'+'{0}'.format(t)+'.jpg'
            # print("热红外路径：", detail)
            #k+=1
           # img1 = cv2.imread(detai3)

            # res1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
           # res1 = cv2.resize(img1,(256,256))
           # thermal_train_set.append(res1)
    #volun.extend(volunteer)
poms_set = np.array(poms_set)    #(5,6)
print("POMS:",poms_set)
print("POMS:",poms_set[0])
# print("PPPPPPPPP",poms_set)

#可见光
print("可见光训练集图片数：",len(train_set))
print("视频数据总数：",n)
train_set=np.array(train_set)
print("TRAIN_SET11",train_set.shape)      #(60,64,64)
train_set = np.reshape(train_set,(n,train_set.shape[0]//n ,train_set.shape[1],train_set.shape[2],3))   # 10,300,64,64,1
print("TRAIN_SET",train_set.shape)       #(60,64,64,1)
train_set = train_set.astype('float32')
print("BBBBBBBB",train_set.shape)
train_set /= 255


#print("热红外训练集图片数：",len(thermal_train_set))
#print("视频数据总数11111：",n)
#thermal_train_set=np.array(thermal_train_set)
#print("TRAIN_SET11",thermal_train_set.shape)     #(60,64,64)
#thermal_train_set = np.reshape(thermal_train_set,(n,thermal_train_set.shape[0]//n,thermal_train_set.shape[1],thermal_train_set.shape[2],3))    #共有796组数据
#print("TRAIN_SET",thermal_train_set.shape)      #(20,3,64,64,3)
#thermal_train_set = thermal_train_set.astype('float32')
#print("CCCCCCCCCC",thermal_train_set.shape)
#thermal_train_set /= 255


# def load_cnn_model():
#     model = load_model('/home/som/lab/ywy/800/_mini_XCEPTION.102-0.66.hdf5')
#     return model

# def predict_score(model,predict_data):
#     scores = model.predict(predict_data, batch_size=16)
#     print(scores)
#     return scores

# def cnn_results_check(cnn_predictions):
#     print("XXXXXXXXXXX",cnn_predictions.shape)         #157*719=112883
#     cnn_predictions = cnn_predictions.reshape(n, cnn_predictions.shape[0] //n, cnn_predictions.shape[1])
#     # cnn_predictions = cnn_predictions.reshape(710, cnn_predictions.shape[0] // 710, cnn_predictions.shape[1])
#     cnn_predictions = np.array(cnn_predictions)
#     return cnn_predictions

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)


def models(poms_set,train_set,thermal_train_set):
    # print("CNNNNNNNNNN",cnn_predictions)


    time_train, time_test, y_train, y_test = train_test_split(thermal_train_set, poms_set, test_size=0.3, random_state=3)
    eeg_train, eeg_test, y_train, y_test = train_test_split(train_set, poms_set, test_size=0.3, random_state=3)
    print("TEST",y_test)
    record1 = pd.DataFrame(y_test, columns=['depression'],
                          )  # index=volun
    for c in record1.columns.values:
        record1.loc[record1[c] > 20, c] = 20
        record1.loc[record1[c] < 0, c] = 0
    #record1.to_csv('/home/som/lab/rongqian/hangtian/model/data/all_data_train_socre_t22.csv')
    print("EEGGGGGG",eeg_train.shape)
    # frequency_train, frequency_test, y_train, y_test = train_test_split(frequencyarr, labelarr, test_size=0.3,
    #                                                                     random_state=3)

    eeginput = Input(shape=(train_set.shape[1], train_set.shape[2],train_set.shape[3],train_set.shape[4]))       #可见光特征(300,64,64,1)
    # print("EGGINPUT:",egginput.shape)
    # input
    timeinput = Input(shape=(thermal_train_set.shape[1], thermal_train_set.shape[2], thermal_train_set.shape[3],thermal_train_set.shape[4]))      #热红外特征

    # frequencyinput = Input(
    #     shape=(frequencyarr.shape[1], frequencyarr.shape[2], frequencyarr.shape[3], frequencyarr.shape[4])) , input_shape=(train_data.shape[1], train_data.shape[2]), name='lstm1'

#CNN提取可见光特征
    regularization = l2(0.001)

    # base
    # img_input = Input(eeginput)
    x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', strides=(1, 1), kernel_regularizer=regularization,
                               use_bias=False))(eeginput)
    x = TimeDistributed(BatchNormalization())(x)
    # x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', strides=(1, 1), kernel_regularizer=regularization,
                               use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    # x = Activation('relu')(x)

    # module 1
    # residual = TimeDistributed(Conv2D(16, (1, 1), strides=(2, 2),
    #                                   padding='same', use_bias=False))(x)
    # residual = TimeDistributed(BatchNormalization())(residual)

    x = TimeDistributed(SeparableConv2D(16, (3, 3), activation='relu', padding='same',
                                        kernel_regularizer=regularization,
                                        use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    # x = Activation('relu')(x)
    x = TimeDistributed(SeparableConv2D(16, (3, 3), padding='same',
                                        kernel_regularizer=regularization,
                                        use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))(x)
    # x = add([x, residual])
    print("XXXXX:",x.shape)

    fc00 = TimeDistributed(Flatten())(x)
    print("FFFFFC0000:",fc00.shape)

    fc0 = TimeDistributed(Dense(units=1024, activation='relu'))(fc00)
    print("FFFFFC0:",fc0.shape)
    #
    #     # output = TimeDistributed(Conv2D(6, (3, 3),
    #     #            # kernel_regularizer=regularization,
    #     #            padding='same'))(x)
    #     print(output.shape)
    #     model = Model(img_input, output)
    #     return model


#LSTM模型
    ls1 = LSTM(256, dropout=0.5)(fc0)
    ls2 = Dense(1024, activation='relu')(ls1)
    ls3 = Dropout(0.5)(ls2)
    ls4 = Dense(1024)(ls3)

#热红外模型
    conv1 = Conv3D(filters=64, kernel_size=(1, 11, 11), padding='same', activation='relu', strides=(1, 4, 4))(timeinput)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(bn1)
    conv2 = Conv3D(filters=96, kernel_size=(1, 5, 5), padding='same', activation='relu', strides=(1, 1, 1))(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 4, 2))(bn2)
    conv3 = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(pool2)
    bn3 = BatchNormalization()(conv3)
    conv4 = Conv3D(filters=64, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(bn3)
    bn4 = BatchNormalization()(conv4)
    conv5 = Conv3D(filters=32, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(bn4)
    bn5 = BatchNormalization()(conv5)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 4, 2))(bn5)
    fla1 = Flatten()(pool3)
    fc1 = Dense(units=1024, activation='relu')(fla1)

    add1 = concatenate([ls4,fc1])

    # fla4 = Flatten()(add1)
    fc4=Dense(units=1024, activation='relu')(add1)
    fc5 = Dense(units=512, activation='relu')(fc4)
    fc6 = Dense(units=256, activation='relu')(fc5)
    fc7 = Dense(units=128, activation='relu')(fc6)
    fc8 = Dense(units=64, activation='relu')(fc7)
    fc9 = Dense(units=32, activation='relu')(fc8)
    fc10 = Dense(units=16, activation='relu')(fc9)
    fc11 = Dense(units=8, activation='relu')(fc10)
    fc12 = Dense(units=6)(fc11)
    # fc13=Dense(units=poms_set)(fc12), activation='relu'

    model = Model(input = [eeginput, timeinput], output=fc12)
    model.summary()

    # model = multi_gpu_model(model, gpus=2)

    model.compile(optimizer=Adam(lr=1e-4), loss=losses.mean_squared_error, metrics=[metrics.MeanSquaredError(),
                                                                                    metrics.MeanAbsoluteError(),
                                                                         metrics.MeanAbsolutePercentageError(),
                                                                         metrics.RootMeanSquaredError(),pearson_r])
    # model.compile(optimizer=Adam(lr=1e-4),loss=losses.mean_squared_error, metrics=[metrics.MeanSquaredError(),
    #                                                             metrics.MeanAbsoluteError(),metrics.MeanSquaredError(),
    #                                                             metrics.MeanAbsolutePercentageError(),
    #                                                             metrics.RootMeanSquaredError()]), pearson_r

    # model_checkpoint = ModelCheckpoint('/home/som/lab/seed-yzj/paper4/model/eeg+videonet.hdf5', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True)

    history = model.fit([eeg_train,time_train], y_train, batch_size=3, epochs=30, verbose=1,
                        validation_data=([eeg_test,time_test], y_test))     #batch_size=32
    model.save('/home/som/lab-data/yuyaodata/depression1.h5')
    sscores = model.predict([eeg_test,time_test], batch_size=1)
    record = pd.DataFrame(sscores, columns=['depression'],
                          )#index=volun
    #for c in record.columns.values:
        #record.loc[record[c] > 20, c] = 20
        #record.loc[record[c] < 0, c] = 0
    #record.to_csv('/home/som/lab/rongqian/hangtian/model/data/all_data_train_socre_t2.csv')


    return sscores,history



if __name__ == '__main__':


    # labelarr,timearr, frequencyarr,eegarr=load_all_data(hr_label_path,imgsdatapath,EEG_folderpath)

    # labelarr=label_preprocessing(labelarr)
    # print(labelarr)
    # timearr, frequencyarr = img_preprocessing(timearr, frequencyarr)
    poms_set = np.array(poms_set)
    # cnn_model = load_cnn_model()
    # cnn_predictions = predict_score(cnn_model, train_set)
    # cnn_predictions = cnn_results_check(cnn_predictions)
    # print("DDDDDDDDDD", cnn_predictions.shape)
    y_prediction,history = models(poms_set,train_set,thermal_train_set)
    print("预测值：", y_prediction)
    # drawlines(history)


















# def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.001):
#     regularization = l2(l2_regularization)
#
#     # base
#     img_input = Input(input_shape)
#     x = TimeDistributed(Conv2D(8, (3, 3),Activation = 'relu', strides=(1, 1), kernel_regularizer=regularization,
#                                             use_bias=False))(img_input)
#     x = TimeDistributed(BatchNormalization())(x)
#     # x = Activation('relu')(x)
#     x = TimeDistributed(Conv2D(8, (3, 3), Activation = 'relu',strides=(1, 1), kernel_regularizer=regularization,
#                                             use_bias=False))(x)
#     x = TimeDistributed(BatchNormalization())(x)
#     # x = Activation('relu')(x)
#
#     # module 1
#     residual = TimeDistributed(Conv2D(16, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False))(x)
#     residual = TimeDistributed(BatchNormalization())(residual)
#
#     x = TimeDistributed(SeparableConv2D(16, (3, 3),Activation = 'relu', padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False))(x)
#     x = TimeDistributed(BatchNormalization())(x)
#     # x = Activation('relu')(x)
#     x = TimeDistributed(SeparableConv2D(16, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False))(x)
#     x = TimeDistributed(BatchNormalization())(x)
#
#     x = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))(x)
#     x = layers.add([x, residual])
#
#     # module 2
#     residual = TimeDistributed(Conv2D(32, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False))(x)
#     residual = TimeDistributed(BatchNormalization())(residual)
#
#     x = TimeDistributed(SeparableConv2D(32, (3, 3),Activation = 'relu', padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False))(x)
#     x = TimeDistributed(BatchNormalization())(x)
#     # x = Activation('relu')(x)
#     x = TimeDistributed(SeparableConv2D(32, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False))(x)
#     x = TimeDistributed(BatchNormalization())(x)
#
#     x = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))(x)
#     x = layers.add([x, residual])
#
#     # module 3
#     residual = TimeDistributed(Conv2D(64, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False))(x)
#     residual = TimeDistributed(BatchNormalization())(residual)
#
#     x = TimeDistributed(SeparableConv2D(64, (3, 3),Activation = 'relu', padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False))(x)
#     x = TimeDistributed(BatchNormalization())(x)
#     # x = Activation('relu')(x)
#     x = TimeDistributed(SeparableConv2D(64, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False))(x)
#     x = TimeDistributed(BatchNormalization())(x)
#
#     x = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))(x)
#     x = layers.add([x, residual])
#
#     # module 4
#     residual = TimeDistributed(Conv2D(128, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False))(x)
#     residual = TimeDistributed(BatchNormalization())(residual)
#
#     x = TimeDistributed(SeparableConv2D(128, (3, 3),Activation = 'relu', padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False))(x)
#     x = TimeDistributed(BatchNormalization())(x)
#     # x = Activation('relu')(x)
#     x = TimeDistributed(SeparableConv2D(128, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False))(x)
#     x = TimeDistributed(BatchNormalization())(x)
#
#     x = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))(x)
#     x = layers.add([x, residual])
#
#     fc0 = TimeDistributed(Dense(units=1024, activation='relu'))(x)
#
#     # output = TimeDistributed(Conv2D(6, (3, 3),
#     #            # kernel_regularizer=regularization,
#     #            padding='same'))(x)
#     print(output.shape)
#     model = Model(img_input, output)
#     return model