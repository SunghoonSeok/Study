from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train[0])
print(y_train[0])
print(len(x_train[0]),len(x_train[11])) # 87 59
print("=======================================")
print(x_train.shape,x_test.shape) # (8982,) (2246,)
print(y_train.shape,y_test.shape) # (8982,) (2246,)

print("뉴스기사 최대길이 : ",max(len(l) for l in x_train)) # 2376
print("뉴스기사 평균길이 : ",sum(map(len,x_train))/(len(x_train))) # 145.5398574927633

# plt.hist([len(s) for s in x_train], bins=50) # 데이터 길이 히스토그램
# plt.show()

# y분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("y분포 : ",dict(zip(unique_elements, counts_elements)))
print("=========================")

# plt.hist(y_train, bins=46) # y 히스토그램
# plt.show()

# x 단어들의 분포
word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index))
print("======================================")

# 키와 벨류를 교체
index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value] = key

# 키 벨류 교환 후
print(index_to_word)
print(index_to_word[1]) # the
print(index_to_word[30979]) # northerly
print(len(index_to_word)) # 30979

# x_train[0]
print(x_train[0])
print(' '.join([index_to_word[index] for index in x_train[0]]))

# y 카테고르 갯수 출력
category = np.max(y_train) + 1 # 46
print("y 카테고리 개수 : ", category)

# y의 유니크한 값 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

############################ 전처리 ################################
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=500)
x_test = pad_sequences(x_test, maxlen=500)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Conv1D

model = Sequential()
model.add(Embedding(10000, 600, input_length=500))
model.add(LSTM(300))
model.add(Dense(46,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=30,mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='min')
file_path = 'c:/data/modelcheckpoint/checkpoint_84.hdf5'
mc = ModelCheckpoint(file_path,monitor='val_acc', save_best_only=True, mode='max',verbose=1)
model.fit(x_train,y_train, batch_size=16, epochs=200, validation_split=0.2, callbacks=[es,lr,mc])



loss, acc = model.evaluate(x_test,y_test,batch_size=16)
print("Loss : ", loss)
print("Accuracy : ", acc)
model2 = load_model('c:/data/modelcheckpoint/checkpoint_84.hdf5')
loss2, acc2 = model2.evaluate(x_test,y_test,batch_size=64)
print("Load_Loss : ", loss2)
print("Load_Accuracy : ", acc2)

# adam
# Loss :  1.9384208917617798
# Accuracy :  0.7150489687919617

# rmsprop
# Loss :  1.9339804649353027
# Accuracy :  0.6892253160476685

# embedding 600, lstm 300
# Loss :  1.691252589225769
# Accuracy :  0.7435441017150879