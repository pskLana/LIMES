import re

f = open("./datasets/Persons1/person12.nt", "r")
trainfile = open("train.txt", "w")
for x in f:
	if '\"' not in x:
		triple = x.split(" ")
		subject = re.sub(r'[<>"]','', triple[0])
		predicate = re.sub(r'[<>"]','', triple[1])
		obj = re.sub(r'[<>"]','', triple[2])
		trainfile.write(subject+"\t"+predicate+"\t"+obj+"\n")
	
trainfile.close()
f.close()