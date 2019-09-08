from scipy.stats import chi2_contingency
import numpy as np
kf_data1 = np.array([[26,147,327],[17,146,337]])
kf_data2 = np.array([[33,6,56,5],[54,14,52,5]])
kf_data3 = np.load('vecs.npy')
re = chi2_contingency(kf_data3)
print(re[0])
print(re[1])
print(re[2])