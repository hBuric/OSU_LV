import numpy as np
import matplotlib . pyplot as plt

x=np.array([3,3,2,1,3])
y=np.array([1,2,2,1,1])   #koordinate tocaka sa slike

plt.plot (x , y , 'g', linewidth =3 , marker =".", markersize =8 )
plt.axis ([0 ,4 ,0 , 4])
plt.xlabel("x-os")
plt.ylabel("y-os")
plt.title("Zadatak 1")
plt.show ()
