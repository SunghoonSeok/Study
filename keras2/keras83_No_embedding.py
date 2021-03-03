from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs =['너무 재밌어요','참 최고에요','참 잘 만든 영화예요',
       '추천하고 싶은 영화입니다.','한 번 더 보고 싶네요','글쎄요',
       '별로에요','생각보다 지루해요','연기가 어색해요','재미없어요',
       '너무 재미없다','참 재밌네요','규현이가 잘 생기긴 했어요']

# 긍정1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre',truncating='post', maxlen=5) # padding이 pre이면 앞, post이면 뒤에 0이 붙음
print(pad_x)
print(pad_x.shape) # (13, 5) maxlen으로 열 조절 가능, 앞에 자를거면 truncating='post', 뒤에 자를거면 'pre'
print(np.unique(pad_x))
print(len(np.unique(pad_x))) # 15가 빠짐

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D
pad_x = pad_x.reshape(13,5,1)
model = Sequential()
# model.add(Embedding(input_dim=28,output_dim=11, input_length=5)) # param# = inputdim*output_dim, 총단어수 *output dim
# model.add(Embedding(28,11)) # None때문에 Flatten에 바로 안먹힘
# model.add(Flatten())
model.add(LSTM(32, input_shape=(5,1)))
# model.add(Dense(20, input_dim=5))
# model.add(Conv1D(32,2))
# model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1]
print(acc)