import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def precision(T, Y):
	tp = T[Y==1].sum()
	fp = np.array(1-T[Y==1]).sum()
	return tp / (tp + fp)

def recall(T, Y):
	tp = T[Y==1].sum()
	fn = T[Y==0].sum()
 	return tp / (tp + fn)

def F1_score(T, Y):	
	if precision==recall==0:
		return 0
	precision = precision(T, Y)
	recall = recall(T, Y)	
	return 2*precision*recall / (precision + recall)