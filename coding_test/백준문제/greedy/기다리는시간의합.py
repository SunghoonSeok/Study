# ATM기 한대로 n명의 사람이 이용해야 할때
# 1번 사람 이용시간 3분, 2번 사람 이용시간 5분, 3번 사람 이용시간 1분이라 할 때,
# 1번 사람의 이용+대기시간은 3분, 2번 사람의 이용+대기시간은 3분+5분=8분, 3번 사람의 이용+대기시간은 3분+5분+1분=9분
# 모든 사람의 총 이용+대기시간은 3분+8분+9분=20분
# 이 시간은 사람들의 이용순서에 따라 달라진다.

num_ppl = int(input())
waiting_time = list(map(int, input().split()))
answer = 0
waiting_time.sort()

for i in range(num_ppl):
    answer += waiting_time[i] * (num_ppl-i)

print(answer)