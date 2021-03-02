from tensorflow.keras.applications import VGG16

# model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
model = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))
# model = VGG16()

model.trainable = False
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

# Loss :  13.56813907623291
# acc :  0.7786999940872192
# 157/157 [==============================] - 3s 16ms/step - loss: 0.7355 - acc: 0.7525
# Best Loss :  0.735488772392273
# Best acc :  0.7524999976158142