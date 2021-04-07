# 기본 입출력
# n = int(input())
# data = list(map(int, input().split()))
# print(n) # 3
# print(data) # [15, 243, 44]
# a, b, c = map(int, input().split())

# 빠르게 입력받기
# import sys
# data = sys.stdin.readline().rstrip()
# print(data)

# 실전에서 유용한 표준 라이브러리

# itertools -> 순열과 조합 라이브러리 (모든 경우의수를 고려해야 하는 경우)
# heapq(우선순위 큐 기능을 구현하기 위해, 최단거리)
# bisect -> 이진탐색
# collections -> 데크, 카운터

# 순열(nPr)
from itertools import permutations
data = ['A', 'B', 'C']
result = list(permutations(data, 3))
print(result)
# [('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), 
# ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')]

# 조합(nCr)
from itertools import combinations
result = list(combinations(data, 2))
print(result) # [('A', 'B'), ('A', 'C'), ('B', 'C')]

# Counter (등장 횟수를 세는 기능 제공)
from collections import Counter
counter = Counter(['r','b','r','g','b','b'])
print(counter['b']) # 3
print(counter['g']) # 1
print(dict(counter)) # {'r': 2, 'b': 3, 'g': 1}

# math(최대공약수, 최소공배수)
import math
# 최대 공약수

print(math.gcd(21,14)) # 7

# 최소공배수
def lcm(a,b):
    return a*b//math.gcd(a,b)
print(lcm(21,14)) # 42