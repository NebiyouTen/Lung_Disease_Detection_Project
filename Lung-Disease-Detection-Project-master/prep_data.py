import os
import xml.etree.ElementTree as ET  
import pandas as pd
import numpy as np
MESH = 17
MAJOR= 0
IMAGE_PATH = 18
MEDLINECITATION = 16
ARTICLE  = 0
ABSTRACT = 2

data_dir = 'ecgen-radiology/'

train_table_name = 'data/cnn_train_table.pkl'
val_table_name   = 'data/cnn_val_table.pkl'


#print(os.listdir(data_dir))
labels = []
image_paths = []
reports = []
data = []

LABLES = {
'normal':0,
'opacity':1,
'cardiomegaly':2,
'calcinosis':3,
'lung/hypoinflation':4,
'calcified granuloma':5,
'thoracic vertebrae/degenerative':6,
'lung/hyperdistention':7,
'spine/degenerative':8,
'catheters, indwelling':9,
'granulomatous disease':10,
'nodule':11,
'surgical instruments':12,
'scoliosis':13,
'osteophyte':14,
'spondylosis':15,
'fractures, bone':16
}

def get_label(text, labels):
	for label in labels:
		if label in text:
			return label
			
	
count = 0;
number_of_reports = len(os.listdir(data_dir))
label_freq = np.zeros(17)
for file_name in os.listdir(data_dir):
	file_name = os.path.join(data_dir, file_name)
	tree = ET.parse(file_name)  
	root = tree.getroot()
	try:	
		img_path = root[IMAGE_PATH].attrib['id']
		image_paths.append(img_path)
		major = root[MESH][MAJOR].text
		img_label = get_label(major.lower(), LABLES.keys())

		if img_label is not None:
			label_freq[LABLES[img_label]]+=1			
			data.append([img_path, LABLES[img_label]])
	except:
		pass
		#print("image doesn't exist")
print(LABLES, label_freq.tolist())
print(np.sum(label_freq))

df = pd.DataFrame(data, columns=['file', 'label'])
train_table = df.sample(frac=0.80) #70-30 split
val_table   = df[~df['file'].isin(train_table['file'])]
print(val_table)

train_table.to_pickle(train_table_name)
val_table.to_pickle(val_table_name)


'''
file_name = os.path.join(data_dir, '484.xml')
tree = ET.parse(file_name)  
root = tree.getroot()
report = []
for elem in root[MEDLINECITATION][ARTICLE][ABSTRACT]:
	#print(elem.text)
	if(elem.text==None):
		report.append('none.')
	else:
		report.append(elem.text)
		
#print(report)
print("\n".join(report))
'''
