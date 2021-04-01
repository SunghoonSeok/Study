# 펠린드롬 문제 -> (소주 만 병만 주소)처럼 문자열이 대칭인지 확인하는 문제

# 1번 풀이
# isalnum() : 영문자, 숫자여부를 판별하는 함수
# lower() : 모두 소문자로 변환
# return 두개로 True False를 구분하고 while문으로 strs가 1개 이하가 될때까지 pop으로 비교 및 제거
s="A man, a plan, a canal: Panama"
t='race a car'
def isPalindrom(s: str) -> bool:
    strs = []
    for char in s:
        if char.isalnum():
            strs.append(char.lower())
    #펠린드롬 여부 판별
    while len(strs) > 1:
        if strs.pop(0) != strs.pop():
            return False
    return True

a = isPalindrom(s=s)
print(a)

# 2번 풀이
# 리스트 대신 collections.deque 사용으로 속도 증가
# pop(0) 보단 popleft()의 속도가 빠르다
import collections
def isPalindrom(s:str) -> bool:
    strs: Deque = collections.deque()

    for char in s:
        if char.isalnum():
            strs.append(char.lower())
    while len(strs) > 1:
        if strs.popleft() !=strs.pop():
            return False
    return True
print(isPalindrom(s=t))

# 정규식으로 불필요한 문자 필터링
import re

def isPalindrom(s:str) -> bool:
    s = s.lower()
    s = re.sub('[^a-z0-9]','',s)
    return s==s[::-1]
