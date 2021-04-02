# 3자리 정수 a,b,c를 곱한 수가 0~9까지의 숫자를 몇개 포함하고 있는지 구하라
# ex) 10030250   -> 0이 4개, 1이 1개, 2가 1개, 3이 1개, 5가 1개이므로
# 4
# 1
# 1
# 1
# 0
# 1
# 0
# 0
# 0
# 0 이런식으로 뽑히도록

a = input()
b = input()
c = input()

number = int(a)*int(b)*int(c)
number = str(number)
for i in range(10):
    print(number.count(str(i)))
