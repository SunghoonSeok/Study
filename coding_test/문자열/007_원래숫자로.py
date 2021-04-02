# 원래 숫자가 있으면 첫째자리수와 둘째자리수를 더한다
# 그리고 원래 숫자의 둘째자리수가 새로운 숫자의 첫째자리
# 더한 숫자의 둘째자리수가 새로운 숫자의 둘째자리로 들어간다
# 이걸 반복해서 얼마만에 원래 수로 돌아오는지 체크
# 예시) 26 -> 68 -> 84 -> 42 -> 26    4회만에 원래 수로 돌아옴

n = int(input())
cycle = 0
temp = n
while True:
    a = temp//10
    b = temp%10
    c = a+b
    d = c%10
    new_num = b*10 + d
    cycle +=1
    if n == new_num:
        break
    else:
        temp = new_num

print(cycle)