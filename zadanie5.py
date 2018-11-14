#!/usr/bin/python3

import string
import re
import math
from urllib.request import urlopen

def maxCount(li):
    maxC = 0
    for l in li:
        if li.count(l) > maxC:
            maxC = li.count(l)
    return maxC

def scal(li, n):
    wynik = {}
    for i in range(n):
        if i*2    in li and i*2+1    in li: wynik[i] = li[i*2]+li[i*2+1]
        if i*2 not in li and i*2+1    in li: wynik[i] = li[i*2+1]
        if i*2    in li and i*2+1 not in li: wynik[i] = li[i*2]
    return wynik
print("czekaj...") 
dokument = ''
listaDoc = []
f = urlopen("http://150.254.36.78/diffs.txt")
lsSyn  = []
ls = [ l.decode('utf-8') for l in f.readlines()]
print("ok, pisz") 
n = int(input())
for i in range(n*2):
    dokument = str(input())
    dokument = dokument.lower()
    listaDoc.append(dokument)
m = int(input())
for i in range(m):
    zapytanie = ((re.sub('[^0-9a-zA-Z]+', ' ',str(input()) )).lower()).split()
    for j in zapytanie:
        for ellis in ls:
            if re.match('^'+j+'\s.*', ellis):
                zapytanie[zapytanie.index(j)] = re.sub('^\S*\s*', '', ellis)
                break
    slownik = {}
    for doc in listaDoc:
        slowa = (re.sub('[^0-9a-zA-Z]+', ' ', doc)).lower()
        slowa = slowa.split()
        for j in slowa:
            for ellis in ls:
                if re.match('^'+j+'\s.*', ellis):
                    slowa[slowa.index(j)] = re.sub('^\S*\s*', '', ellis)
                    break
        ilWDoc = 0
        ilWCalosci = 0
        for zap in zapytanie:
            ilWDoc += slowa.count(zap)
            if slowa.count(zap) == 0:
                ilWDoc = 0
                ilWCalosci = 1
                break
            ilWCalosci += maxCount(slowa)
        tf = ilWDoc / ilWCalosci
        if listaDoc.index(doc)%2 == 0:
            tf = tf*2
        slownik[listaDoc.index(doc)]=tf
    slownik = {key: value for key, value in slownik.items() if value != 0}
    if len(slownik) != 0:
        idf = math.log(len(listaDoc) / len(slownik))
    else : idf = 0
    for k, s in slownik.items():
        slownik[k]= s * idf
    slownik = scal(slownik, n)
    print(sorted(slownik, key=slownik.get, reverse=True))
