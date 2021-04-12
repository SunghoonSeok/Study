# def solution(s):
#     answer = ''
#     if len(s)%2 ==0:
#         answer = s[len(s)//2-1]+s[len(s)//2]
#     else:
#         answer = s[len(s)//2]
        
#     return answer
# print(solution('qwer'))

def solution(s):
    answer = ''
    if len(s)%2 ==0:
        a = (len(s)//2)
        answer = s[a-1:a+1]
        print(answer)
    else:
        answer = s[len(s)//2]
        
    return answer
print(solution('qwer'))