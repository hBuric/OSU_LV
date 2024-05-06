List = []

while(1):
    user_input = input()
    if(user_input == "Done"):
        break
    try:
        num = float(user_input)
        List.append(num)
    except ValueError:
        print("Wrong input")
        continue

total = 0
for num in List:
    total += num

print("Numbers count: " + str(len(List)) + " average: " + str(total/len(List)) + " min: " + str(min(List)) + " max: " + str(max(List)))

List.sort()
print(List)
