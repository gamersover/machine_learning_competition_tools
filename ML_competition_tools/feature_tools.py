import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def one_hot_encode(df, one_hot_features):
	le = LabelEncoder()
	oe = OneHotEncoder()

	one_hots = []
	for feat in one_hot_features:
		labels = le.fit_transform(df[feat])
		one_hot = oe.fit_transform(labels.reshape(-1, 1))
		one_hots.append(one_hot)

	return sparse.hstack(one_hots)


class BinLabelEncoder:
	"""
	Attention: avoid overfit! shouldn't fit transform train and val simultaneously
	BinLabel doing this:  
	                   feature label endoder                  groupby mean
		feature label       ------->        feature label       ------>     feature  label
		    1.1    2                           0     2                           0    14/3
		    4.2    3                           1     3                           1     7/2
		    3.2    4 						   1     4
		    1.7    5                           0     5
		    2.6    7                           0     7

	and use the relation between feature and label to transform the other feature in val or test dataset

	"""
	def __init__(self, bins, **params):
		self.bins = bins


	def fit(self, x, y):
		self.df = pd.DataFrame({"feature": x, "label": y})
		self.df["feature"] = pd.cut(self.df["feature"], bins=self.bins, labels=range(self.bins))
		self.bin2mean = self.df.groupby("feature")["label"].mean()
		return self

	def transform(self, x):
		self.df["feature"] = self.df["feature"].apply(lambda i: self.bin2mean[i])
		return self.df["feature"].values

	def fit_transform(self, x, y):
		self.fit(x, y)
		return self.transform(x)


class LabelBinEncoder:
	"""
	LabelBin: feature should be discrete, label should be continuous
	LabelBin doing this:  
	                   label <label endoder>                  groupby mean
		feature label       ------->        feature label       ------>     feature  label
		    0    2                           0        0                        0      2/3
		    1    3                           1        0                        1      2/2
		    0    4 						     0        1
		    0    5                           0        1
		    1    7                           1        2
	"""
	def __init__(self, bins, **params):
		self.bins = bins

	def fit(self, x, y):
		self.df = pd.DataFrame({"feature": x, "label": y})
		self.df["label"] = pd.cut(self.df["label"], bins=self.bins, labels=range(self.bins)).astype(int)
		self.bin2mean = self.df.groupby("feature")["label"].mean()

	def transform(self, x):
		self.df["feature"] = self.df["feature"].apply(lambda i: self.bin2mean[i])
		return self.df["feature"].values

	def fit_transform(self, x, y):
		self.fit(x, y)
		return self.transform(x)





