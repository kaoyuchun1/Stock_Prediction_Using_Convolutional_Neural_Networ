
#%% import

import os
import pandas as pd
import numpy as np
from sklearn.metrics import  confusion_matrix
import datetime
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
import TA4
import copy
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

#%% IO setting 
    
dataname="TWSE.xlsm"   #基金data 名稱
datapath="J:\\730新金融商品部\\人工智慧小組\\高宇駿\\台指"

#%%

start_time = time.time()

#setting inputs 

index1=["D(9)", "K(9)","RSI(5)","RSI(10)","W%R(5)","W%R(10)"]
index2=['日期','收盤價']
index3=['日期','收盤價',"D(9)", "K(9)","RSI(5)","RSI(10)","W%R(5)","W%R(10)"]
index4=['日期','淨值']
index5=['日期','收盤價',"最高價", '最低價','成交量']
    #standard 
Bstd=["min","min","min","min","max","max"]
Sstd=["max","max","max","max","min","min"]

    #kmean range
Kmeanmin=3
Kmeanmax=11 #kmeanmax +1
sellscore=-1
    #period for data validation 
#validation_end_day = datetime.date(2019,7,9)
validation_end_day = datetime.date.today()+datetime.timedelta(days=1)
train_end_day =validation_end_day-relativedelta(years=5)
validation_start_day = train_end_day-timedelta(days=1)
train_start_day =train_end_day-relativedelta(years=1)

std = StandardScaler()

#####################

Rolling_days=20

days=5

#####################

#%%

#read excel file
output=list()
rawdata= pd.read_excel(datapath+'\\'+dataname,header=4)
rawdata['日期']=rawdata['日期'].dt.date

TrainTest=pd.DataFrame(rawdata , columns = index5)


#%% Calculate Technical indicators

TrainTest['Max']=TrainTest['收盤價'][::-1].rolling(Rolling_days).max()
TrainTest['Min']=TrainTest['收盤價'][::-1].rolling(Rolling_days).min()
TrainTest['DMax']=TrainTest['Max'].diff(-1)
TrainTest['DMin']=TrainTest['Min'].diff(-1)


    #RSI 

for i in range(6,21):
    TrainTest=TA4.RSI(TrainTest,i)

    #W%R 
    
for i in range(6,21):
    TrainTest=TA4.willian(TrainTest,i)
    
    #SMA
    
for i in range(6,21):
    TrainTest=TA4.SMA(TrainTest,i,0)    

    #EMA
    
for i in range(6,21):
    TrainTest=TA4.EMA(TrainTest,i,0)

    #WMA
    
for i in range(6,21):
    TrainTest=TA4.WMA(TrainTest,i,0)

    #HMA
    
for i in range(6,21):
    TrainTest=TA4.HMA(TrainTest,i)
    
    #TEMA
    
for i in range(6,21):
    TrainTest=TA4.TEMA(TrainTest,i)

    #CCI
    
for i in range(6,21):
    TrainTest=TA4.CCI(TrainTest,i)

    #CMO
    
for i in range(6,21):
    TrainTest=TA4.CMO(TrainTest,i)
    
    #MACD
    
for i in range(6,21):
    N1=7+i
    N2=20+i
    TrainTest=TA4.MACD(TrainTest,N1,N2)    
    
    #PPO
    
for i in range(6,21):
    N1=7+i
    N2=20+i
    TrainTest=TA4.PPO(TrainTest,N1,N2)    
    
    #ROC
    
for i in range(6,21):
    TrainTest=TA4.ROC(TrainTest,i)     
    
    #CMFI
    
for i in range(6,21):
    TrainTest=TA4.CMFI(TrainTest,i)    

    #DMI
    
for i in range(6,21):
    TrainTest=TA4.DMI(TrainTest,i)    

    #SAR
    
for i in range(6,21):
    TrainTest=TA4.SAR(TrainTest,i)        


#%% Labeling

count=0

TrainTest['Label']='H'

for i in range(0,len(TrainTest['DMin']-Rolling_days)):
    
    if TrainTest['DMin'][i]==0:
        count+=1
    elif  count>=days :        
        Min=TrainTest['Min'][i]        
        for j in range(Rolling_days):            
            if TrainTest['收盤價'][i+j]==Min:                
                TrainTest['Label'][i+j]='B'                
        count=0        
    else:        
        count=0

count=0

for i in range(0,len(TrainTest['DMax'])-Rolling_days):
    
    if TrainTest['DMax'][i]==0:
        count+=1
    elif  count>=days :        
        Max=TrainTest['Max'][i]        
        for j in range(Rolling_days):            
            if TrainTest['收盤價'][i+j]==Max:                
                TrainTest['Label'][i+j]='S'
        count=0         
    else:
        count=0 
  
raise SystemExit(0)
#%%  split data for training test and validation

TrainTest2 = copy.copy(TrainTest) 

#validation = TrainTest2[TrainTest2['日期']> validation_start_day ]
#TrainTest2 = TrainTest2[(TrainTest2['日期'] < train_end_day) & (TrainTest2['日期'] > train_start_day)]

validation = TrainTest2[(TrainTest2['日期'] < train_end_day) & (TrainTest2['日期'] > train_start_day)]
TrainTest2 = TrainTest2[TrainTest2['日期']> validation_start_day ]

#raise SystemExit(0)

#Normalization

label_va=validation['Label'].reset_index(drop=True)
date_va =validation['日期'].reset_index(drop=True)
validation=validation.drop(['Label'],axis=1)
validation=validation.drop(['日期'],axis=1)

label_tr=TrainTest2['Label'].reset_index(drop=True)
date_tr =TrainTest2['日期'].reset_index(drop=True)
TrainTest2=TrainTest2.drop(['Label'],axis=1)
TrainTest2=TrainTest2.drop(['日期'],axis=1)

std = StandardScaler()
TrainTest2=std.fit_transform(TrainTest2)
validation=std.transform(validation)

TrainTest2=pd.DataFrame(TrainTest2)
validation=pd.DataFrame(validation)

TrainTest2=pd.concat([date_tr,TrainTest2,label_tr],axis=1,ignore_index=False)
validation=pd.concat([date_va,validation,label_va],axis=1,ignore_index=False)

TrainTest2.columns=TrainTest.columns
validation.columns=TrainTest.columns

raise SystemExit(0)

#%%  plot 
'''
#x= range(len(R1),0,-1)
date = TrainTest['日期'][TrainTest['日期'] > train_start_day]

label_B = validation[validation['Label']=='B']
label_S = validation[validation['Label']=='S']

label_B_train = TrainTest2[TrainTest2['Label']=='B']
label_S_train = TrainTest2[TrainTest2['Label']=='S']

fig , (ax1,ax2) = plt.subplots(2, 1, sharex=False, figsize=(10,60))
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.3, wspace=0.3, hspace=0.3)
ax1.set_title('TWSE(test)')
ax1.plot(validation['日期'],validation['收盤價'],label="TWSE",color = 'steelblue' )
ax1.plot(label_B['日期'],label_B['收盤價'],'b.',label="Buy")
ax1.plot(label_S['日期'],label_S['收盤價'],'r.',label="Sell")
ax1.grid(ls='--')
ax1.legend()

ax2.set_title('TWSE(train)')  
ax2.plot(TrainTest2['日期'],TrainTest2['收盤價'],label="TWSE",color = 'steelblue')
ax2.plot(label_B_train['日期'],label_B_train['收盤價'],'b.',label="Buy")
ax2.plot(label_S_train['日期'],label_S_train['收盤價'],'r.',label="Sell")
ax2.grid(ls='--')
ax2.legend()

plt.show()
'''
#%%

TrainTest2['Label'][TrainTest2['Label']=='B']=-1
TrainTest2['Label'][TrainTest2['Label']=='H']= 0
TrainTest2['Label'][TrainTest2['Label']=='S']= 1

validation['Label'][validation['Label']=='B']=-1
validation['Label'][validation['Label']=='H']= 0
validation['Label'][validation['Label']=='S']= 1

################################################
#增加BS 樣本洗回去
 
l1_new = pd.DataFrame()
l2_new = pd.DataFrame()
for idx, row in TrainTest2.iterrows():
    if row[-1] == -1:
        for i in range(20):
            l1_new = l1_new.append(row)
    if row[-1] == 1:
        for i in range(20):
            l2_new = l2_new.append(row)

TrainTest2 = TrainTest2.append(l1_new)
TrainTest2 = TrainTest2.append(l2_new)

# shuffle  洗回去
TrainTest2 = shuffle(TrainTest2)
###############################################

Label_train = TrainTest2['Label']
Label_test  = validation['Label']

drop_index=['日期', '收盤價', '最高價', '最低價', '成交量', 'Max', 'Min', 'DMax', 'DMin','Label'] 

TrainTest2 =  TrainTest2.drop(drop_index,axis=1)
validation =  validation.drop(drop_index,axis=1)

TrainTest2.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)
Label_train.reset_index(drop=True, inplace=True)
Label_test.reset_index(drop=True, inplace=True)

#%%

from keras.models import Sequential

from keras.layers import Dense, Dropout , Activation , Flatten

from keras.layers import Conv2D , MaxPooling2D , ZeroPadding2D 

import keras

def reverse_one_hot(predictions):
    reversed_x = []
    for x in predictions:
        reversed_x.append(np.argmax(np.array(x)))
    return reversed_x

#%% CNN pre

test_labels = keras.utils.to_categorical(Label_test, 3)
train_labels = keras.utils.to_categorical(Label_train, 3)

train_images = TrainTest2.as_matrix()
test_images = validation.as_matrix()

train_images = train_images.reshape(train_images.shape[0], 15, 15, 1)
test_images = test_images.reshape(test_images.shape[0], 15, 15, 1)

#train_images = TrainTest2.reshape(TrainTest2.shape[0], 15, 15, 1)
#test_images = validation.reshape(validation.shape[0], 15, 15, 1)

#%%  CNN 模型建立 

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(15,15,1),activation='relu',padding='same'))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))

model.add(Dropout(0.25))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(1280, activation='relu'))

model.add(Dense(2560, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(5120, activation='relu'))

#model.add(Dense(10240, activation='relu'))

#model.add(Dense(20480, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(3,activation='softmax'))


#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),\
                metrics=['accuracy', 'mae', 'mse'])

model.fit(train_images, train_labels, 128, 100, verbose=1,validation_split=0.2,validation_data=None)

#train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation=0.2,epochs=200,batch_size=300,verbose=2)

predictions = model.predict(test_images, batch_size=128, verbose=1)
print(model.evaluate(test_images, test_labels, batch_size=128, verbose=1))

print("Train conf matrix: \n", confusion_matrix(np.array(reverse_one_hot(train_labels)),
                                              np.array(reverse_one_hot(model.predict(train_images,128, verbose=1)))))

print("Test conf matrix: \n",  confusion_matrix(np.array(reverse_one_hot(test_labels)),
                                              np.array(reverse_one_hot(predictions))))

print(classification_report(np.array(reverse_one_hot(test_labels)), np.array(reverse_one_hot(predictions)), labels=[0, 1, 2]))


#%%  plot 
'''
label_pre= ['H']*len(test_labels)
label_re = reverse_one_hot(test_labels)
for i in range(0,len(test_labels)):
    if label_re[i]==1:
        label_pre[i]='B'
    elif label_re[i]==2:
        label_pre[i]='S'
        
label_pre=pd.DataFrame(label_pre)
label_pre.columns=['Label']
#x= range(len(R1),0,-1)
date = TrainTest['日期'][TrainTest['日期'] > validation_start_day]
validation_pre=label_pre.append(date)

label_B = validation[validation['Label']=='B']
label_S = validation[validation['Label']=='S']

label_B_train = TrainTest2[TrainTest2['Label']=='B']
label_S_train = TrainTest2[TrainTest2['Label']=='S']

fig , (ax1,ax2) = plt.subplots(2, 1, sharex=False, figsize=(10,60))
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.3, wspace=0.3, hspace=0.3)
ax1.set_title('TWSE(test)')
ax1.plot(validation['日期'],validation['收盤價'],label="TWSE",color = 'steelblue' )
ax1.plot(label_B['日期'],label_B['收盤價'],'b.',label="Buy")
ax1.plot(label_S['日期'],label_S['收盤價'],'r.',label="Sell")
ax1.grid(ls='--')
ax1.legend()

ax2.set_title('TWSE(train)')  
ax2.plot(TrainTest2['日期'],TrainTest2['收盤價'],label="TWSE",color = 'steelblue')
ax2.plot(label_B_train['日期'],label_B_train['收盤價'],'b.',label="Buy")
ax2.plot(label_S_train['日期'],label_S_train['收盤價'],'r.',label="Sell")
ax2.grid(ls='--')
ax2.legend()

plt.show()

'''

'''
    #%% Performace Evaluation # strategy return
    
R1 = PE.R1(TrainTest) #DataFrame include date&nav
R1=R1.reset_index(drop=True)
R2,cost2,asset2,gain2 = PE.R2(TrainTest,1000)
R4_TWSE,cost4_TWSE,gain4_TWSE,unrealgain4_TWSE = PE.R4(TrainTest,1000,1000,0)
        
    #%%

x= range(len(R1),0,-1)
weekday = rawdata['日期'][rawdata['日期'] > last_day]

fig , (ax1,ax2,ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,60))
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.3, wspace=0.3, hspace=0.3)
ax1.set_title('return(%)')
#ax1.plot(weekday,R1,label="Passive (R1)")
ax1.plot(weekday,R2,label="Periodic(R2)",color = 'steelblue' )
#ax1.plot(weekday,R4,label="Robo-Advisor(R4)",color = 'darkorange')
ax1.plot(weekday,R4_TWSE,label="Robo-Advisor(R4_TWSE)",color = 'silver')
ax1.grid(ls='--')
ax1.legend()

ax2.set_title('Cost')  
ax2.plot(weekday,C2,label="Periodic(C2)",color = 'steelblue')
#ax2.plot(weekday,C4,label="Robo-Advisor(C4)",color = 'darkorange')
ax2.plot(weekday,C4_TWSE,label="Robo-Advisor(C4_TWSE)",color = 'silver')
ax2.grid(ls='--')
ax2.legend()

ax3.set_title('Unrealized Gain')
ax3.plot(weekday,G2,'c',label="Un Gain2",color = 'steelblue')
ax3.plot(weekday,pd.DataFrame(R4_TWSE)[:]+pd.DataFrame(UR4_TWSE)[:],'r',label="Robo-Advisor(TG4_TWSE)",color = 'silver')
ax3.grid(ls='--')
ax3.legend()
plt.show()

   
end_time = time.time()
elapsed = end_time - start_time
print("time taken" , elapsed , "seconds.")
print("time taken" , elapsed/60 , "minutes.")
print("time taken" , elapsed/3600 , "hours.")    
#print('R1:',R1[0],'\nR2:',R2[0],'\nR4:',R4.iloc[0,0],'\nR4_TWSE:',R4_TWSE.iloc[0,0])
#print('UnrealGain2:',gain2[0],'\nunrealgain4:',unrealgain4[0])
'''

