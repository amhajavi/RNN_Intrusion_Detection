import os

data = []
flag = []
for file_name in os.listdir('Train'):
	for file in os.listdir(os.path.join('Train',file_name)):
		with open(os.path.join('Train',file_name, file)) as data_record:
			record = data_record.readlines()[0].strip().split(' ')
			data.append((record, [1,0]))
			if 'Attack' in file_name:
				flag.append([1,0])
			else:
				flag.append([0,1])
print flag