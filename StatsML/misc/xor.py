import numpy as np
import matplotlib.pyplot as plt
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.structure import  LinearLayer,SigmoidLayer

N_wej = 2
N_hid = 10
N_wyj = 1

ZU = SupervisedDataSet(N_wej,N_wyj)
ZU.addSample((0,0),(0.0,))
ZU.addSample((0,1),(1.0,))
ZU.addSample((1,0),(1.0,))
ZU.addSample((1,1),(0.0,))

siec = buildNetwork(N_wej, N_hid, N_wyj, outclass=LinearLayer)

trener = BackpropTrainer(siec, ZU, learningrate=0.1, lrdecay=1.0, momentum=0.9, verbose=False, batchlearning=False, weightdecay=0.002)
# dla sigmoidLayer dobre learningrate=0.5 momentum=0.9

#trener = RPropMinusTrainer(siec, etaminus=0.5, etaplus=1.2, deltamin=1e-06, deltamax=5.0, delta0=0.1)
#trener.setData(ZU)

n = 500
blad = np.zeros(n)
wagi = np.zeros((n, len(siec.params)))

plt.ion()
plt.subplot(211)
l_err, = plt.plot(blad)
plt.ylim([0, 0.3])
plt.subplot(212)
l_wagi = plt.plot(wagi)
plt.ylim([-5,5])

for i in range(n):
    blad[i] = trener.train()
    wagi[i,:] = siec.params
    l_err.set_ydata(blad)
    for k in range(len(l_wagi)):
        l_wagi[k].set_ydata(wagi[:,k])
    plt.draw()

for ex in ((0,0), (0,1), (1,0), (1,1)):
    print ex, '-->', siec.activate(ex)
