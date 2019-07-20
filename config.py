# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:06:28 2019

@author: Mou
"""

workspace = "/home/mouwang/Work/ABSE"
data_dir = "/home/mouwang/Work/ABSE/mini_data"
  
magnification = 2

Tr_SNR = 0
Te_SNR = 0

sample_rate = 16000
fft_size = 512
hop_size = 256
window_type = 'hamming'
n_window = 512      # windows size for FFT
n_overlap = 256     # overlap of window

n_concat = 7
n_hop = 3
lr = 1e-4       # learning rate
n_hid = 2048

cuda = True
visualize = False

bgn_iter=0
fin_iter=10001
interval_iter=1000

iteration = 10000
