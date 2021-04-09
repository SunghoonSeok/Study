SET @hour := -1; -- 변수 선언

SELECT (@hour := @hour + 1) as HOUR,
(SELECT COUNT(*) FROM ANIMAL_OUTS WHERE HOUR(DATETIME) = @hour) as COUNT
FROM ANIMAL_OUTS
WHERE @hour < 23

-- 시간이 23시까지 데이터가 없어서 시간을 변수로 지정하여 0부터 23까지 만들어준다
-- 변수를 이용하여 뽑아내는 Hour과 전체적으로 한번 select from where을 거쳐야만 뽑을 수 있는 count를 같이 내줌
-- count(*)모든 수를 뽑는데, animal_outs에서, hour(datetime)이 변수에서 뽑아내는 hour과 같은 것을
