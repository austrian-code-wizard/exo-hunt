from cmath import nan
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
# fpath = 'HD142527_16May15_Line_80pctcut_REAL_a21m2s1iwa4_hp2.0-KLmodes-all.fits'
fpath = 'V47_20150518045046802513.fits'


data = fits.getdata(fpath)
torched = torch.from_numpy(data.astype(np.float32))
print(torched.shape)
print(data.shape)

# pool = nn.MaxPool1d(3, stride=3)

# torched = torch.from_numpy(data.astype(np.float32)).transpose(0, 2)

# print(torched.shape)

# split = torch.split(torched, 1, dim=2)
# shape = torched.shape[0] // 2
# split = torch.split(torched, (shape, shape + 1), dim=0)
# megasplit = []
# for elem in split:
#     split2 = torch.split(elem, (shape, shape + 1), dim=1)
#     megasplit.append(split2)

# for elem in megasplit:
#     for spl in elem:
#         for tp in torch.split(spl, 1, dim=2):
#             plt.figure()
#             plt.imshow(tp, cmap='gray')
#             plt.colorbar()
#             plt.show()


# for img in split:
plt.figure()
plt.imshow(torched, cmap='gray')
plt.colorbar()
plt.show()

# maxxed = pool(torched)
# print(maxxed.shape)
# unique = np.unique(data)
# print(data.shape)
# print(unique.shape)