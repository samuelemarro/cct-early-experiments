# Generate two random numbers with the following conditions:
# 1. They must be two-digit numbers (i.e. >= 10, <= 99)
# 2. Their sum must be less than 100
# 3. The sum of the second digit of the first number and the second digit of the second number must be less than 10
# 4. If the digits of the second number are switched, the new sum must be less than 100 [dropped]
# 5. All digits must be distinct
# 6. 0 cannot appear in any position

import random

def generate_numbers():
    while True:
        number_1 = random.randint(10, 99)
        number_2 = random.randint(10, 99)
        cond_2 = number_1 + number_2 < 100
        cond_3 = number_1 % 10 + number_2 % 10 < 10
        #number_2_switched = int(str(number_2)[::-1])
        #cond_4 = number_1 + number_2_switched < 100
        cond_4 = True
        cond_5 = len(set(str(number_1) + str(number_2))) == 4
        cond_6 = '0' not in str(number_1) + str(number_2)

        if cond_2 and cond_3 and cond_4 and cond_5 and cond_6:
            return number_1, number_2
        
for i in range(7):
    print(generate_numbers())