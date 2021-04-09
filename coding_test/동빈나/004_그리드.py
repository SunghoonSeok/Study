# 거스름돈
# N원의 거스름돈을 500원, 100원, 50원, 10원으로 거슬러주기
# 최소의 동전만 사용하자
# 그러면 그냥 큰거부터 쭈루룩하면 돼(그리드)

# 그렇게 해도 되는 이유
# 가지고 있는 동전 중에서 큰단위가 항상 작은 단위의 배수이므로
# 작은 단위의 동전들을 종합해 다른 해가 나올 수 없기 때문이다.

n= int(input())
count = 0
array = [500, 100, 50, 10]
for coin in array:
    count += n//coin
    n = n%coin
print(count)

# 1로 만들기
# N에서 1을 빼주기, N을 K로 나눔 이 두가지만 써서 1로 만들어라

# N에 대하여 최대한 많이 나누기를 수행하면 된다.
# N의 값을 줄일 때 2 이상의 수로 나누는 작업이 1을 빼는 것보다 많이 줄일 수 있다.
n, k = map(int, input().split())

result = 0
while True:
    target = (n//k) * k
    result += (n-target)
    n= target
    if n<k:
        break
    result += 1
    n //= k
result += (n-1)
print(result)    

# 곱하기 혹은 더하기
# 가장 큰수를 만들어라
# 대부분의 경우 '+' 보다는 'x'가 더 값을 크게 만든다.
# 다만 두 수중 하나라도 '0' 혹은 '1'이라면 덧셈이 낫겠지?

data = input()
# 첫번째 문자를 숫자로 변경하여 대입
result = int(data[0])

for i in range(1, len(data)):
    # 두 수 중 하나라도 '0' 혹은 '1' 인 경우, 더하기 수행
    num = int(data[i])
    if num <= 1 or result <= 1:
        result += num
    else:
        result *= num
print(result)
