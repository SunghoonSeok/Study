from tensorflow.keras.datasets import imdb, reuters
import matplotlib.pyplot as plt
import numpy as np
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=1000
)

# [실습/과제] Embedding으로 모델 만들것!!

print(x_train[0])
print(y_train[0])
print(len(x_train[0]),len(x_train[11])) # 218 99
print("=======================================")
print(x_train.shape,x_test.shape) # (25000,) (25000,)
print(y_train.shape,y_test.shape) # (25000,) (25000,)

print("뉴스기사 최대길이 : ",max(len(l) for l in x_train)) # 2494
print("뉴스기사 평균길이 : ",sum(map(len,x_train))/(len(x_train))) # 238.71364

# plt.hist([len(s) for s in x_train], bins=50) # 데이터 길이 히스토그램
# plt.show()

# y분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("y분포 : ",dict(zip(unique_elements, counts_elements)))
print("=========================")

# plt.hist(y_train, bins=46) # y 히스토그램
# plt.show()
'''
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
'''

############################ 전처리 ################################
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Conv1D

model = Sequential()
model.add(Embedding(1000, 400, input_length=100))
model.add(LSTM(200))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=30,mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='min')
file_path = 'c:/data/modelcheckpoint/checkpoint_85.hdf5'
mc = ModelCheckpoint(file_path,monitor='val_acc', save_best_only=True, mode='max',verbose=1)
model.fit(x_train,y_train, batch_size=16, epochs=200, validation_split=0.2, callbacks=[es,lr,mc])


loss, acc = model.evaluate(x_test,y_test,batch_size=16)
print("Loss : ", loss)
print("Accuracy : ", acc)
model2 = load_model('c:/data/modelcheckpoint/checkpoint_85.hdf5')
loss2, acc2 = model2.evaluate(x_test,y_test,batch_size=16)
print("Load_Loss : ", loss2)
print("Load_Accuracy : ", acc2)

# Loss :  1.541557788848877
# Accuracy :  0.8256000280380249

# Load_Loss :  0.36793699860572815
# Load_Accuracy :  0.8389599919319153