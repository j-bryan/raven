import numpy as np
import matplotlib.pyplot as plt
from generators import toFile

np.random.seed(12345)

def generateVARMA(mean, cov, ar, ma, nObs, burnin=100):
  """
    Generate a VARMA time series
  """
  nVars = len(mean)
  signal = np.zeros((nObs + burnin, nVars))
  noise = np.random.multivariate_normal(np.zeros(nVars), cov, size=nObs+burnin)

  # initialize the first few values using just the noise terms, before we can use the full AR model
  signal[:len(ar)] = mean + noise[:len(ar)]
  for i in range(len(ar), nObs + burnin):
    signal[i] = mean \
                + np.sum([arj @ signal[i-j-1] for j, arj in enumerate(ar)], axis=0) \
                + np.sum([maj @ noise[i-j-1] for j, maj in enumerate(ma)], axis=0) \
                + noise[i]
    for j, arj in enumerate(ar):
      signal[i] += arj @ signal[i-j-1]

  signal = signal[burnin:]
  return signal

# We'll generate two VAR(1) signals with different parameters
mean1 = np.array([2, -2])
cov1 = np.array([[1.0, 0.8],
                 [0.8, 1.0]])
ar1 = np.array([[[0.4, 0.1],
                 [-0.1, 0.4]]])
ma1 = np.array([])
print(np.linalg.eigvals(ar1[0]))
signal1 = generateVARMA(mean1, cov1, ar1, ma1, nObs=200)

mean2 = np.array([-2, 2])
cov2 = np.array([[0.5, 0.3],
                 [0.3, 2.0]])
ar2 = np.array([[[-0.4, 0.1],
                 [0.1, 0.2]]])
ma2 = np.array([])
print(np.linalg.eigvals(ar2[0]))
signal2 = generateVARMA(mean2, cov2, ar2, ma2, nObs=200)

# Concatenate the two signals
signal = np.r_[signal1, signal2]
signal = np.hstack((np.arange(len(signal)).reshape(-1, 1), signal))

toFile(signal, 'VARMAClustered', pivotName='seconds')

plt.plot(signal[:, 1])
plt.plot(signal[:, 2])
plt.show()
