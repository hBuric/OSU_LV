words = []
words_dict = {}
counter = 0

with open('song.txt') as fhand:
    for line in fhand:
        line = line.rstrip()
        words += line.split()  

for word in words:
    if word in words_dict:
        words_dict[word] += 1
    else:
        words_dict[word] = 1

for word in words_dict:
    if words_dict[word] == 1:
        counter += 1

print(words_dict)
print("Broj riječi koje se pojavljuju samo jednom:", counter)
