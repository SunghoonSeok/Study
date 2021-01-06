# 애완동물을 소개해 주세요~
animal = "강아지"
name = "연탄이"
age = 4
hobby = "산책"
is_adult = age >= 3

print("우리집 "+animal+ "의 이름은 "+ name+"예요")
print(name + "는 " + str(age) + "살이며, "+hobby+ "을 아주 좋아해요")
print(name+ "는 어른일까요? "+ str(is_adult))

animal = "고양이"
name = "해피"
age = 2
hobby = "공놀이"

print("우리집 "+animal+ "의 이름은 "+ name+"예요")
# print(name + "는 " + str(age) + "살이며, "+hobby+ "을 아주 좋아해요")
print(name, "는 ", age , "살이며, ", hobby ,"을 아주 좋아해요")
print(name+ "는 어른일까요? "+ str(is_adult))

# str 써야하는 상황 -> 숫자를 글자화 해야할 때
# 변수와 글자를 혼용할 때, +대신 ,를 쓸 수 있다.