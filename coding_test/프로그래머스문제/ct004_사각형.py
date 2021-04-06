def solution(w,h):
    answer = 0
    for i in range(1,w):
        a= int(h(1-i*(1/w)))
        answer = answer + a
    return answer

print(solution(8, 12))