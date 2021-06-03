import cv2
cap = cv2.VideoCapture('C:/video/eight_acoustic.mp4')

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('C:/video/eight_acoustic2.mp4', fourcc, 23.98, (1280,720))
count = 0
while True: # 무한 루프
    ret, frame = cap.read() # 두 개의 값을 반환하므로 두 변수 지정

    if not ret: # 새로운 프레임을 못받아 왔을 때 braek
        break
    
    # 정지화면에서 윤곽선을 추출
    edge = cv2.Canny(frame, 50, 150)
    
    inversed = ~frame  # 반전

    if ret:
        grey_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        invert = cv2.bitwise_not(grey_img)
        blur = cv2.GaussianBlur(invert,(21,21),0)
        invertedblur = cv2.bitwise_not(blur)
        sketch = cv2.divide(grey_img,invertedblur,scale=256.0)

        
        cv2.imwrite("C:/video/sketch/eight_acoustic/frame%d.jpg" % count, sketch)
        cv2.imshow('edge', sketch)
        count += 1
        
    if cv2.waitKey(10) == 27:
        break



# 작업 완료 후 해제
out.write(sketch)
cap.release()
out.release()
cv2.destroyAllWindows()