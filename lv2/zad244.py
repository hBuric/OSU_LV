import numpy as np
import matplotlib.pyplot as plt

black=np.zeros((50, 50))
white=np.ones((50, 50))
x=np.hstack((black,white))  #slazemo
y=np.hstack((white,black))  #odgovarajuci
box = np.vstack((x,y))      #oblik
plt.imshow(box, cmap='gray')
plt.show()
