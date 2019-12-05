# Kalman.py
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import time


datax = loadmat(r'C:\Users\HP\Desktop\TP01_kalman\etat_cache')
datay = loadmat(r'C:\Users\HP\Desktop\TP01_kalman\observation')
print(datax.keys())
x = datax['x']
y = datay['y']
print(x)
print(np.shape(x))
n=100
#parameteres
c = 0.9
alpha = -math.pi/8
e = 0.1
H = np.eye(2)
R = np.array([[math.cos(alpha),-math.sin(alpha)],[math.sin(alpha),math.cos(alpha)]])

#Les variables 
xtilde= np.zeros((2,100));
k = np.zeros((2,2,100));
P_ = np.zeros((2,2,100));
Ptilde = np.zeros((2,2,100));
x_ = np.zeros((2,100));
w = np.random.randn(2,100); 

#initialisation
x_[0:2,0] = [0,0]
Ptilde[:,:,0] = np.array([[1,0],[0,1]])
Qv = [[1,0],[0,1]]
print(np.shape(x[:,10]))

# Kalman filter
for i in range(2,100,1):
	x_[:,i] = c*np.dot(R,x[:,(i-1)])+math.sqrt(1-c**2)*w[:,i]

	P_[:,:,i] = np.linalg.multi_dot([np.asarray(c)*R,P_[:,:,i-1],np.transpose(np.asarray(c)*R)]) + np.asarray(e**2)*Qv

	#Kalman gain
	k[:,:,i] = P_[:,:,i]*np.transpose(H)*np.linalg.inv(H*P_[:,:,i]*np.transpose(H)+np.asarray(e**2)*Qv)

	xtilde[:,i] = x_[:,i] + np.dot(k[:,:,i],(y[:,i]-np.dot(H,x_[:,i])))

	Ptilde[:,:,i] = (np.eye(2)-np.dot(k[:,:,i],H))*np.linalg.inv(P_[:,:,i]);

sum = 0;
for i in range(n):
	sum = sum + ((xtilde[:,i]-x[:,i])**2);

print(sum)
plt.figure(1)
plt.subplot(211)
plt.plot(x[0],'b-',label = 'etat caché 0')
plt.plot(xtilde[0],'r',label = 'estimation 0')
plt.legend()
plt.subplot(212)
plt.plot(x[1],'b-',label = 'etat caché 1')
plt.plot(xtilde[1],'r',label = 'estimation 1')
plt.legend()
plt.show()

plt.figure(2)
axes = plt.gca()
axes.set_xlim(-3,3)
axes.set_ylim(-3,3)
line, = axes.plot([], [], 'b-')

for i in range(100):
	plt.plot(x[0,i],x[1,i],'b-',marker = '*')
	plt.plot(xtilde[0,i],xtilde[1,i],'r-',marker = '+')
	plt.plot([x[0,i],x[0,i+1]],[x[1,i],x[1,i+1]],'b-')
	plt.plot([xtilde[0,i],xtilde[0,i+1]],[xtilde[1,i],xtilde[1,i+1]],'r-')
	plt.plot()
	plt.pause(1e-17)
	time.sleep(0.01)
plt.show()
	


