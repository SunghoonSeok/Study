# N개의 화폐 단위로 K원을 딱 맞추는 문제
# 다만 화폐를 최소 갯수만 사용할것
# 만약 화폐 갯수가 3개면 1, 5, 10원만 사용 가능 N<=10, K<=100,000,000(N,K는 자연수)

#동전과 값 입력받기
n,k=map(int,input().split())
#동전을 입력받을 리스트 초기화하기
coin=[]

#동전 종류 입력받기
for i in range(n):
  coin.append(int(input()))

#동전개수 셀 변수 초기화
result=0

#동전 내림차순으로 정렬하기
coin.sort(reverse=True)

#동전 개수 구하기
for i in coin:
  if k==0: break
  result+=k//i
  k%=i
  

print(result)