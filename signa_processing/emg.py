import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

def filtruj_gornie(syg,Fs=128.):
	f=np.zeros_like(syg)
	b,a=ss.butter(3,0.5/(Fs/2),btype="highpass")
	for i in range(len(syg)):
		f[i]=ss.filtfilt(b,a,syg[i])
	return f

def filtruj_50(syg,Fs=128.):
	f=np.zeros_like(syg)
	b,a=ss.butter(2,[48/(Fs/2),52/(Fs/2)],btype="bandstop")
	for i in range(len(syg)):
		f[i]=ss.filtfilt(b,a,syg[i])
	return f

def widmo(syg, Fs = 128.):
	f=np.fft.fft(syg)
	fr=np.fft.fftfreq(len(syg),1./Fs)
	wid=np.abs(f)**2
	return np.array([fr, wid])

emg=np.fromfile("Sygnal.raw", "float32").reshape((-1,4)).T

emg = filtruj_gornie(emg, Fs = 1024.)
emg = filtruj_50(emg, Fs = 1024.)

swobodny = emg[0][:58000]
napiety = emg[0][58000:58000*2]
swobodny_widmo = widmo(swobodny, Fs=1024.)[1]
swobodny_czestotliwosci = widmo(swobodny, Fs=1024.)[0]
napiety_widmo = widmo(napiety, Fs = 1024.)[1]
napiety_czestotliwosci = widmo(napiety, Fs = 1024.)[0]

print np.mean(swobodny)
print np.std(swobodny)
print np.mean(napiety)
print np.std(napiety)
print np.mean(np.abs(swobodny))
print np.mean(np.abs(napiety))

plt.subplot(2,2,1)
plt.plot(swobodny)
plt.subplot(2,2,2)
plt.plot(swobodny_czestotliwosci, swobodny_widmo)
plt.subplot(2,2,3)
plt.plot(napiety)
plt.subplot(2,2,4)
plt.plot(napiety_czestotliwosci, napiety_widmo)
plt.show()
