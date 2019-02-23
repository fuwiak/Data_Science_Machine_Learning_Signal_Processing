import requests
from bs4 import BeautifulSoup


data = []

soup = BeautifulSoup(data.text, 'html.parser')

words = []

for p in soup.findAll('p'):
	temp = p.text.split(" ")
	for w in temp:
		words.append(w)
