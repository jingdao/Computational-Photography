import numpy as np
import matplotlib.pyplot as plt

features_train = np.load('features_train_fc7_pr.npy')
labels_train =  np.load('labels_train_fc7_pr.npy')

real = []
fake = []

for i in range(len(labels_train)):
	if labels_train[i] > 0:
		real.append(features_train[i])
	else:
		fake.append(features_train[i])

real = np.mean(real,0)
fake = np.mean(fake,0)

plt.figure()
plt.subplot(2,1,1)
plt.plot(real,'g')
plt.title('Mean feature values for real vs fake images (fc7)')
plt.subplot(2,1,2)
plt.plot(fake,'r')
plt.show()
