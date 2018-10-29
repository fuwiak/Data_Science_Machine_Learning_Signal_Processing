def periodogram(syg,fs,N,window,war=False):
    fv,fS = widmo(syg*window(len(syg)),fs,N,war=False)
    return fv,fS

def widmo(syg,fs,N,war=False):
    ft = np.fft.fft(syg,N)
    fv = np.fft.fftfreq(len(ft),1/fs)
    #ft = np.fft.fftshift(ft)
    #fv = np.fft.fftshift(fv)
    if(war==True):  #Odcięcie ujemnych cżęstości
        ind = np.where(fv<0)[0][0]
        ft = ft[:ind]
        fv = fv[:ind]
    else:  # Ustawienie częstości od -nyq do nyq
        ft= np.fft.fftshift(ft)
        fv = np.fft.fftshift(fv)
    return fv,abs(ft)

def welch(sig, w, fs, L, ov):
        """
        L - dlugosc okna
        ov - przykrycie okien w %
        """
        window = {'hamming':np.hamming, 'blackman':np.blackman}[w]
        i = L
        step = (1-ov)*L
        N = len(sig)
        ft = np.zeros(L)
        no_per = 0
        fv = np.zeros(len(sig))
        while i < N:
            s_ = sig[i-L:i]
            fv_loc, ft_ = periodogram(s_,fs,len(s_),window=window)
            ft += ft_
            i += step
            no_per += 1
            fv = fv_loc
        return fv, ft/no_per, no_per

def impuls(t0,T):
    "t0 - miejsce z jedynką, T - czas trwania."
    t = py.linspace(0,T,100*T,endpoint=False)
    imp = np.zeros(len(t))
    imp[int(100*t0)]=1
    return t,imp

def widmo_okienkowane(syg,fs,N,window,war=False):
    fv,fS = widmo(syg*window(len(syg)),fs,N,war=False)
    return fv,fS
def czestosci_fazy(syg,fs,N,window,war=False):
    fv,fS = widmo_okienkowane(syg,fs,N,window,war=False)
    ft = np.fft.fft(syg,N)
    cz = []
    tab = np.arange(2,len(fS)-2)
    for i in tab:
        if (fS[i] > fS[i-1] > fS[i-2]) and (fS[i] > fS[i+1] > fS[i+2]) and fS[i]>1:
            cz.append(fv[i])
    moc = []
    angle = []
    for i in range(len(cz)):
        ind = np.where(fv==cz[i])[0][0]
        angle.append(np.angle((ft[ind])))
        moc.append(abs(ft)[ind]**2)
    return cz,angle,moc
