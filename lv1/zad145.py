file = open("SMSSpamCollection.txt")
SPAM_word_count = []
HAM_word_count = []
SPAM_escalamtion_mark_count = 0
for line in file:
    line = line.rstrip()
    if line.startswith('h'):
        HAM_message = line[4:]
        HAM_word_count.append(len(HAM_message.split()))
    elif line.startswith('s'):
        SPAM_message = line[5:]
        SPAM_word_count.append(len(SPAM_message.split()))
        if SPAM_message[-1] == '!':
            SPAM_escalamtion_mark_count -= -1

print(f"Average word count for SPAM: {sum(SPAM_word_count)/len(SPAM_word_count)}")
print(f"Average word count for HAM: {sum(HAM_word_count)/len(HAM_word_count)}")
print(f"SPAM messages that end with exclamation mark: {SPAM_escalamtion_mark_count}")
file.close()
