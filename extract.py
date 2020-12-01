import glob
import numpy as np

opt_cutoff = {'karate':14, 'football':94, 'jazz':158, 'email':594, 'delaunay':703,'netscience':899, 'power':2203,'as-22july06':3303,'hep-th':3926,'star2':4542,'star':6902,'dummy1':2, 'dummy2':3}

record_time, record_nV = {}, {}
for fn in glob.glob('./{}_output/*'.format('SA')):

    fin = open(fn, 'r').readlines()
    name = fn.split('/')[-1].split('_')[0]
    try:
        time, nV = fin[-1].split()[0].split(',')
    except:
        print (fn)
        continue

    time, nV = float(time), float(nV)
    if name not in record_time:
        record_time[name] = [time]
        record_nV[name] = [nV]
    else:
        record_time[name].append(time)
        record_nV[name].append(nV)

for key in record_time.keys():
    opt_nv = opt_cutoff.get('{}'.format(key), 1e-15)
    err = (np.mean(record_nV[key]) - opt_nv)/opt_nv
    print ('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(key, 
                                   np.mean(record_time[key]), np.std(record_time[key]), 
                                   np.mean(record_nV[key]), np.std(record_nV[key]), err, opt_nv))
# import pdb;pdb.set_trace()