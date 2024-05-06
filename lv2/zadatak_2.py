import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data.csv", delimiter=',')
data = data[1:]
data = data.transpose()

#a
print("a) " + str(len(data)))
print("--------------------------------------------------")

#b
fig, axs = plt.subplots(2)
axs[0].scatter(data[1], data[2], s=2)
axs[0].set_xlabel('visina')
axs[0].set_ylabel('masa')
axs[0].set_title('scatter (visina vs masa)')

#c
axs[1].scatter(data[1][::50], data[2][::50], s=5)
axs[1].set_xlabel('visina')
axs[1].set_ylabel('masa')
axs[1].set_title('scatter (visina vs masa)')

#d
print("Max visina: " + str((data[1]).max()))
print("Min visina: " + str(min(data[1])))
print("Avg visina: " + str((data[1]).mean()))
print("--------------------------------------------------")

#e
data = data.transpose()
mask = data[:,0] == 1
men = data[mask]
men = men.transpose()

print("[Man]Max visina: " + str((men[1]).max()))
print("Min visina: " + str(min(men[1])))
print("Avg visina: " + str((men[1]).mean()))
print("--------------------------------------------------")
mask = data[:,0] == 0
women = data[mask]
women = women.transpose()
print("[Woman]Max visina: " + str((women[1]).max()))
print("Min visina: " + str(min(women[1])))
print("Avg visina: " + str((women[1]).mean()))

plt.tight_layout()
plt.show()
