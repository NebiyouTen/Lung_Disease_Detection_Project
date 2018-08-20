import os
import xml.etree.ElementTree as ET  
import pandas as pd

MESH = 17
MAJOR= 0
IMAGE_PATH = 18
MEDLINECITATION = 16
ARTICLE  = 0
ABSTRACT = 2

data_dir = 'ecgen-radiology/'
train_table_name = 'data/train_table.pkl'
val_table_name   = 'data/val_table.pkl'


#print(os.listdir(data_dir))
labels = []
image_paths = []
reports = []
data = []


for file_name in os.listdir(data_dir):
	file_name = os.path.join(data_dir, file_name)
	tree = ET.parse(file_name)  
	root = tree.getroot()
	try:	
		img_path = root[IMAGE_PATH].attrib['id']
		image_paths.append(img_path)
		major = root[MESH][MAJOR].text
		if(major.lower()=='normal'):
			label = 0
		else:#abnormal
			label = 1
		report = []
		for elem in root[MEDLINECITATION][ARTICLE][ABSTRACT]:
			if(elem.text==None):
				report.append('none.')
			else:
				report.append(elem.text)

		report_str = "\n".join(report)
		data.append([img_path, report_str, label, major])
	except:
		pass
		#print("image doesn't exist")

df = pd.DataFrame(data, columns=['file', 'report', 'label','caption'])
train_table = df.sample(frac=0.80) #70-30 split
val_table   = df[~df['file'].isin(train_table['file'])]

train_table.to_pickle(train_table_name)
val_table.to_pickle(val_table_name)

print(val_table) 

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
