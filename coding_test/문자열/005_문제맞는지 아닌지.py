# 문제가 맞으면  Yes 아니면 No를 출력
li = input().split()
a = int(li[0])
b = int(li[2])
c = int(li[4])
if a + b == c:
    print("YES")
else:
    print("NO")