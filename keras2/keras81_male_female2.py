# 이미지는
# data/image/vgg 에 4개 넣으시오
# 개 고양이 라이언 슈트
# 파일명
# dog1.jpg, cat1.jpg, lion1.jpg, suit1.jpg

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_face1 = load_img('../data/image/my_face/1/my_face1.jpg', target_size=(224,224))
img_face2 = load_img('../data/image/my_face/1/my_face2.jpg', target_size=(224,224))
img_face3 = load_img('../data/image/my_face/1/my_face3.PNG', target_size=(224,224))
img_face4 = load_img('../data/image/my_face/1/my_face4.jpg', target_size=(224,224))
img_face5 = load_img('../data/image/my_face/1/my_face5.jpg', target_size=(224,224))
img_face6 = load_img('../data/image/my_face/1/my_face6.PNG', target_size=(224,224))


arr_face1 = img_to_array(img_face1)
arr_face2 = img_to_array(img_face2)
arr_face3 = img_to_array(img_face3)
arr_face4 = img_to_array(img_face4)
arr_face5 = img_to_array(img_face5)
arr_face6 = img_to_array(img_face6)

# RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_face1 = preprocess_input(arr_face1)
arr_face2 = preprocess_input(arr_face2)
arr_face3 = preprocess_input(arr_face3)
arr_face4 = preprocess_input(arr_face4)
arr_face5 = preprocess_input(arr_face5)
arr_face6 = preprocess_input(arr_face6)
arr_input =np.stack([arr_face1,arr_face2,arr_face3,arr_face4,arr_face5,arr_face6])
print(arr_input.shape) # (4, 224, 224, 3)


# 2. 모델 구성
model = VGG16()
results = model.predict(arr_input)

print(results)
print('results.shape : ', results.shape) # results.shape :  (4, 1000)

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions
results = decode_predictions(results, top=3)
print("===========================================")
print("dog1 : ", results[0])
print("===========================================")
print("cat1 : ", results[1])
print("===========================================")
print("lion1 : ", results[2])
print("===========================================")
print("suit1 : ", results[3])
print("===========================================")
# ===========================================
# dog1 :  [('n02111889', 'Samoyed', 0.6032323), ('n02114548', 'white_wolf', 0.32809937), ('n02104029', 'kuvasz', 0.01873776)]
# ===========================================
# cat1 :  [('n02124075', 'Egyptian_cat', 0.59878814), ('n02123045', 'tabby', 0.22296095), ('n02123159', 'tiger_cat', 0.12437381)]
# ===========================================
# lion1 :  [('n03291819', 'envelope', 0.2933515), ('n04548280', 'wall_clock', 0.09355649), ('n02708093', 'analog_clock', 0.050705183)]       
# ===========================================
# suit1 :  [('n04509417', 'unicycle', 0.5397687), ('n03623198', 'knee_pad', 0.0700843), ('n04019541', 'puck', 0.066398285)]
# ===========================================