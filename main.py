
#%% import data 
import time
start_time = time.time()
import pandas as pd

#df=pd.read_csv('J:\\730新金融商品部\\人工智慧小組\\高宇駿\\台指\\TWSE.csv')
df=pd.read_csv('TWSE.csv')

col1=['日期','開盤價','最高價','最低價','收盤價','成交量']

df=df[col1]
#%% data preprocessing

from sklearn.preprocessing import StandardScaler

col=['開盤價','最高價','最低價','收盤價','成交量']

import copy
df1=copy.copy(df)
scaler = StandardScaler()

scaler.fit(df[col])
df[col]=scaler.transform(df[col])

#%% labelling  N= 幾天後股價
label=list()
N=20
for i in range(len(df)-1,N-1,-1):
    if df['收盤價'].iloc[i]>df['收盤價'].iloc[i-N]:
        label.append(1)
    else:
        label.append(0)
        
for i in range(N):
    label.append(float('nan'))

df['label']=label[::-1]

df=df.dropna(axis=0)
df.reset_index(inplace=True,drop=True)

#%% append  N2=幾天內資料
N2=20

df_re=pd.DataFrame()

for i in range(len(df)-N2):
    df_list=df.iloc[i:i+N2,1:6].values.reshape(1,N2*5)
    df_re=df_re.append(pd.DataFrame(df_list))

df_re.reset_index(inplace=True,drop=True)

df_re['label']=df['label'][0:len(df_re)]
df_re['日期']=df['日期'][0:len(df_re)]

#%% 
import datetime

df_re['日期']=pd.to_datetime(df['日期'])
df_re['日期']=df_re['日期'].dt.date

df_va = df_re[df_re['日期']> datetime.date(2012,12,31) ]
df_re = df_re[~(df_re['日期']> datetime.date(2012,12,31))]

#raise SystemExit(0)

date_va=df_va['日期']

df_re.pop('日期')
df_va.pop('日期')
#df_va.pop('label')


#%%  訓練測試

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( df_re.iloc[:,:-1],df_re['label'], test_size=0.2,random_state=100)
#X_train, X_test, y_train, y_test = train_test_split( df_re.iloc[:,:-1] ,df_re['label'], test_size=0.2)

#%% CNN pre

import keras
import numpy as np

test_labels = keras.utils.to_categorical(y_test, 2)
train_labels = keras.utils.to_categorical(y_train, 2)
va_labels = keras.utils.to_categorical(df_va.iloc[:,-1], 2)

df_va.pop('label')

train_images = X_train.as_matrix()
test_images   = X_test.as_matrix()
va_images    = df_va.as_matrix()

train_images = train_images.reshape(train_images.shape[0], N2, 5,1)
test_images = test_images.reshape(test_images.shape[0], N2, 5,1)
va_images  = va_images.reshape(va_images.shape[0], N2, 5,1)

def reverse_one_hot(predictions):
    reversed_x = []
    for x in predictions:
        reversed_x.append(np.argmax(np.array(x)))
    return reversed_x

#raise SystemExit(0)
#%% 建模

from keras.models import Sequential

from keras.layers import Dense, Dropout , Activation , Flatten

from keras.layers import Conv2D , MaxPooling2D , ZeroPadding2D 

from keras.layers.advanced_activations import LeakyReLU

import keras.callbacks as callbacks

#%%

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(N2,5,1),activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))

model.add(LeakyReLU(alpha=0.1))

#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(filters=640,kernel_size=(3,3),padding='same'))
#
#model.add(LeakyReLU(alpha=0.1))
#
#model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
#
#model.add(LeakyReLU(alpha=0.1))
#
#model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
#
#model.add(LeakyReLU(alpha=0.1))

model.add(Flatten())

model.add(Dense(2000, activation='relu'))

model.add(Dropout(0.8))

model.add(Dense(1000, activation='relu'))

model.add(Dense(2,activation='softmax'))

earlyStopping = callbacks.EarlyStopping(monitor='val_loss',patience=20)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),\
#                metrics=['accuracy', 'mae', 'mse'])

model.fit(train_images, train_labels, 1000, epochs=1000, verbose=1, callbacks=[earlyStopping],validation_split=0.2)

#train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation=0.2,epochs=200,batch_size=300,verbose=2)

predictions = model.predict(test_images, batch_size=128, verbose=1)

#print(model.evaluate(test_images, test_labels, batch_size=128, verbose=1))

from sklearn.metrics import  confusion_matrix

print("Train conf matrix: \n", confusion_matrix(np.array(reverse_one_hot(train_labels)),
                                              np.array(reverse_one_hot(model.predict(train_images,128, verbose=1)))))

print("Test conf matrix: \n",  confusion_matrix(np.array(reverse_one_hot(test_labels)),
                                              np.array(reverse_one_hot(predictions))))

from sklearn.metrics import classification_report

print(classification_report(reverse_one_hot(test_labels),reverse_one_hot(predictions), labels=[0, 1]))

import winsound
winsound.Beep(300,2000)

end_time =time.time()
elapsed = end_time - start_time 

print("time taken", elapsed , "seconds")

#%%  存model
'''
import pickle #pickle模块

with open('save/CNNmodel.pickle', 'wb') as f:
    pickle.dump(model, f)
'''
#%% prediction 

predictions_va = model.predict(va_images, batch_size=128, verbose=1)

print("Test conf matrix: \n",  confusion_matrix(reverse_one_hot(va_labels),
                                              reverse_one_hot(predictions_va)))

print(classification_report(reverse_one_hot(va_labels), reverse_one_hot(predictions_va), labels=[0,1]))

predictions_reva = np.array(reverse_one_hot(predictions_va))
predictions_reva=pd.DataFrame(predictions_reva)

'''
label_va=list()

for i in range (len(predictions_va)):
    L=predictions_va[i][0]-predictions_va[i][1]
    if L>0.8:
        label_va.append(0)
    elif L<-0.8:
        label_va.append(1)
    else :
        label_va.append(2)
'''

#predictions_reva=pd.DataFrame(label_va)

predictions_reva['日期']=date_va

df1['日期']=pd.to_datetime(df1['日期'])
df1['日期']=df1['日期'].dt.date

df2=copy.copy(df1)

df1=df1[~(df1['日期']<predictions_reva['日期'].iloc[-1])]
df1=df1[~(df1['日期']>predictions_reva['日期'].iloc[0])]

df2=df2.head(len(predictions_reva))


predictions_reva.columns=['label','日期']
#predictions_reva.reset_index(drop=True)
df1=df1.reset_index(drop=True)

#%% plot

import matplotlib.pyplot as plt

label_B = df1[predictions_reva['label']==0]
label_S = df1[predictions_reva['label']==1]

label_B_re = df2[predictions_reva['label']==0]
label_S_re = df2[predictions_reva['label']==1]

weekday = date_va

fig , (ax1 , ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,40))
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.3, wspace=0.3, hspace=0.3)
ax1.set_title('Pre_TWSE')
ax1.plot(weekday,df1['收盤價'],label="TWSE",color = 'steelblue' )
ax1.plot(label_B['日期'],label_B['收盤價'],'b.',label="Buy")
ax1.plot(label_S['日期'],label_S['收盤價'],'r.',label="Sell")
ax1.grid(ls='--')
ax1.legend()

ax2.set_title('Re_TWSE')
ax2.plot(weekday,df1['收盤價'],label="TWSE",color = 'steelblue' )  
ax2.plot(label_B_re['日期'],label_B_re['收盤價'],'b.',label="Buy")
ax2.plot(label_S_re['日期'],label_S_re['收盤價'],'r.',label="Sell")
#ax2.plot(weekday,C4_TWSE,label="Robo-Advisor(C4_TWSE)",color = 'silver')
ax2.grid(ls='--')
ax2.legend()
