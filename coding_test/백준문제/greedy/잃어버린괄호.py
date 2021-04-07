# 5자리 이하의 수와 + - 괄호만 사용하여 식을 입력
# 괄호를 다 지움 -> 이게 입력값인듯
# 식의 길이는 50 이하
# 괄호를 쳐서 최소의 수 만들기

# - 기준으로 괄호치면 최소값이 된다.
# 55 - 40 + 20 + 30 - 20 + 30 의 최소값은
# 55 - (40 + 20 + 30) - (20 + 30) 이런 식인 것

a = input().split('-')
num = []
for i in a:
    cnt = 0
    s = i.split('+')
    for j in s:
        cnt += int(j)
    num.append(cnt)
n = num[0]
for i in range(1, len(num)):
    n -= num[i]
print(n)

arr = input().split('-') 
s = 0 
for i in arr[0].split('+'): 
    s += int(i) 
for i in arr[1:]: 
    for j in i.split('+'):
            s -= int(j) 

print(s)
