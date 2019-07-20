# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:45:17 2019

@author: Mou
"""

import os
import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt

import config


def plot_training_stat():
    """Plot training and testing loss. 
    
    Args: 
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      bgn_iter: int, plot from bgn_iter
      fin_iter: int, plot finish at fin_iter
      interval_iter: int, interval of files. 
    """
    workspace = config.workspace
    tr_snr = config.Tr_SNR
    bgn_iter = config.bgn_iter
    fin_iter = config.fin_iter
    interval_iter = config.interval_iter

    tr_losses, te_losses, iters = [], [], []
    
    # Load stats. 
    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    for iter in range(bgn_iter, fin_iter, interval_iter):
        stats_path = os.path.join(stats_dir, "%diters.p" % iter)
        dict = pickle.load(open(stats_path, 'rb'))
        tr_losses.append(dict['tr_loss'])
        te_losses.append(dict['te_loss'])
        iters.append(dict['iter'])
        
    # Plot
    line_tr, = plt.plot(tr_losses, c='b', label="Train")
    line_te, = plt.plot(te_losses, c='r', label="Test")
    plt.axis([0, len(iters), 0, max(tr_losses)])
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(handles=[line_tr, line_te])
    plt.xticks(np.arange(len(iters)), iters)
    plt.show()
    
    
def calculate_pesq(data_type):
    """Calculate PESQ of all enhaced speech. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of clean speech. 
      te_snr: float, testing SNR. 
    """
    workspace = config.workspace
    data_dir = config.data_dir
    
    speech_dir = os.path.join(data_dir,'{}_speech'.format(data_type))

    te_snr = config.Te_SNR
    
    # Remove already existed file. 
    os.system('rm _pesq_itu_results.txt')
    os.system('rm _pesq_results.txt')
    
    # Calculate PESQ of all enhaced speech. 
    enh_speech_dir = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr))
    names = os.listdir(enh_speech_dir)
    for (cnt, na) in enumerate(names):
        print(cnt, na)
        enh_path = os.path.join(enh_speech_dir, na)
        
        speech_na = na.split('.')[0]
        speech_path = os.path.join(speech_dir, "%s.WAV" % speech_na)
        
        # Call executable PESQ tool. 
        cmd = ' '.join(["./pesq", speech_path, enh_path, "+16000"])
        os.system(cmd) 
    
def get_stats():
    """Calculate stats of PESQ. 
    """
    pesq_path = "_pesq_results.txt"
    with open(pesq_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
        
    pesq_dict = {}
    for i1 in range(1, len(lis) - 1):
        li = lis[i1]
        na = li[0]
        pesq = float(li[1])
        noise_type = na.split('.')[1]
        if noise_type not in pesq_dict.keys():
            pesq_dict[noise_type] = [pesq]
        else:
            pesq_dict[noise_type].append(pesq)
        
    avg_list, std_list = [], []
    f = "{0:<16} {1:<16}"
    print(f.format("Noise", "PESQ"))
    print("---------------------------------")
    for noise_type in pesq_dict.keys():
        pesqs = pesq_dict[noise_type]
        avg_pesq = np.mean(pesqs)
        std_pesq = np.std(pesqs)
        avg_list.append(avg_pesq)
        std_list.append(std_pesq)
        print(f.format(noise_type, "%.2f +- %.2f" % (avg_pesq, std_pesq)))
    print("---------------------------------")
    print(f.format("Avg.", "%.2f +- %.2f" % (np.mean(avg_list), np.mean(std_list))))

   
    
if __name__ == '__main__':
    
#    plot_training_stat()
    
    data_type = 'test'
    calculate_pesq(data_type)
    get_stats()