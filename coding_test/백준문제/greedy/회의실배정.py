# 하나의 회의실에서 여러번의 회의를 할때,
# 회의의 수를 최대 몇 번까지 가능할지 구하는 문제
# 회의마다 시작시간, 종료시간이 존재하며, 회의가 겹치게 할 수는 없다.

# 회의가 빨리 끝날수록 많은 회의를 집어 넣을 수 있으므로
# 빨리 끝나는 순서대로 오름차순 정렬을 진행
# 하지만 이대로라면 끝나는 시간이 같은 요소들의 정렬이 문제가 됨.
# 그렇기에 시작하는 시간의 오름차순도 진행

import sys 
N = int(sys.stdin.readline()) # 입력
time = [[0]*2 for _ in range(N)] 
for i in range(N): 
    s, e = map(int, sys.stdin.readline().split()) 
    time[i][0] = s # 시작시간
    time[i][1] = e # 종료시간
time.sort(key = lambda x: (x[1], x[0])) # 종료시간 정렬후 종료시간이 같다면 시작시간 정렬

# 정렬이 끝났으므로 제일 빨리 끝나는 회의를 넣고 그 회의의 종료시간을 end_time 변수에 저장한다.
cnt = 1 
end_time = time[0][1]

# 정렬이 된 상태이므로 종료시간 보다 늦거나 같게 시작하는 회의를 넣고 
# 그 회의의 종료시간을 end_time에 저장 이후 반복 
for i in range(1, N): 
    if time[i][0] >= end_time: 
        cnt += 1 
        end_time = time[i][1] 

print(cnt)

def matzip(name, food):
    go(name)
    eat(food)
    if food == good:
        matzip_list.append(name)
    else:
        pass