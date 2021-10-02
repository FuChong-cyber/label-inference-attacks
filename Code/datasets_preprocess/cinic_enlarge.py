"""
https://github.com/BayesWatch/cinic-10/blob/master/notebooks/enlarge-train-set.ipynb
"""
import glob
import os
from shutil import copyfile

cinic_directory = "D:/Datasets/CINIC10"
enlarge_directory = "D:/Datasets/CINIC10-L"
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
sets = ['train', 'valid', 'test']
if not os.path.exists(enlarge_directory):
    os.makedirs(enlarge_directory)
if not os.path.exists(enlarge_directory + '/train'):
    os.makedirs(enlarge_directory + '/train')
if not os.path.exists(enlarge_directory + '/test'):
    os.makedirs(enlarge_directory + '/test')

for c in classes:
    if not os.path.exists('{}/train/{}'.format(enlarge_directory, c)):
        os.makedirs('{}/train/{}'.format(enlarge_directory, c))
    if not os.path.exists('{}/test/{}'.format(enlarge_directory, c)):
        os.makedirs('{}/test/{}'.format(enlarge_directory, c))

for s in sets:
    for c in classes:
        source_directory = '{}/{}/{}'.format(cinic_directory, s, c)
        filenames = glob.glob('{}/*.png'.format(source_directory))
        for fn in filenames:
            dest_fn = fn.split('/')[-1]
            if s == 'train' or s == 'valid':
                dest_fn = '{}/train/{}'.format(enlarge_directory, dest_fn)
                copyfile(fn, dest_fn)
            elif s == 'test':
                dest_fn = '{}/test/{}'.format(enlarge_directory, dest_fn)
                copyfile(fn, dest_fn)
        print("----{} done.".format(c))
    print("--{} done".format(s))
print("[All done]")
