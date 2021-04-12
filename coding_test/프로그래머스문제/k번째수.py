

def solution(array, commands):
    answer = []
    temp=[]
    for num in range(len(commands)):
        temp = array[commands[num][0]-1:commands[num][1]]
        temp.sort()
        print(temp)
        a = temp[commands[num][2]-1]
        answer.append(a)
    return answer

array = [1, 5, 2, 6, 3, 7, 4]
commands = 	[[2, 5, 3], [4, 4, 1], [1, 7, 3]]

print(solution(array, commands))