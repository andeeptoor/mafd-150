import pandas as pd
import os
import numpy as np
import urllib2
import re
from bs4 import BeautifulSoup

dataset_file = '../dataset.csv'

df = pd.read_csv(dataset_file)

i = 1
total = len(df)
for _, row in df.iterrows():
	if i % 30:
		df.to_csv(dataset_file, index=False)
	print "[%d/%d]" % (i,total)
	i += 1
	url = row['url'].replace('!HD','').replace('.jpg','')
	url = re.sub('[(]\d[)]', '', url)
	if not 'wikiart' in url:
		continue

	if not row['title'] is np.nan:
		continue

	if '/images' in url:
		base = url.split('/images')[-1]
	else:
		base = '/' + url.split('/')[-1]
	image_url = 'https://www.wikiart.org/en' + base
	print image_url

	try:
		page = urllib2.urlopen(image_url)
	except:
		continue

	soup = BeautifulSoup(page, 'html.parser')
	info = soup.find('div', attrs={'class':'info'})
	row['title'] = info.find('h1').text.strip().encode('unicode-escape').decode('utf-8')
	style = soup.find(text="Style:")
	if style is None:
		continue
	row['category'] = style.findNext('span').text.strip().encode('unicode-escape').decode('utf-8')
print df
df.to_csv(dataset_file, index=False)
	# print info.find('div', attrs={'class':'info-line'})