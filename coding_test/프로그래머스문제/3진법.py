def solution(n):
    answer = ''
    result = 0

    while n >= 1:
        rest = n % 3
        n //= 3
        answer += str(rest)

    i = 0
    for idx in range(len(answer)-1, -1, -1):
        result += int(answer[idx]) * (3**i)
        i += 1

    return result

print(solution(45))