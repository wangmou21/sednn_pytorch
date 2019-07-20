# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:20:08 2019

@author: Mou
"""


import os
import h5py
import numpy as np

import config


from utils import create_folder,read_audio,normalize

def calc_sp(audio, fft_size, hop_size, window):
    
    sp = stft.stft(x=audio, 
                    window_size=fft_size, 
                    hop_size=hop_size, 
                    window=window, 
                    mode='complex')

    sp = sp.astype(np.complex64)

    return sp

def calculate_features():
    workspace = config.workspace    
    data_dir = config.data_dir
    tr_speech_dir = os.path.join(data_dir,"train_speech")
    tr_noise_dir = os.path.join(data_dir,"train_noise")
    te_speech_dir = os.path.join(data_dir,"test_speech")
    te_noise_dir = os.path.join(data_dir,"test_noise")
     
    sample_rate = config.sample_rate
    fft_size = config.fft_size
    hop_size = config.hop_size
    window_type = config.window_type
        
    # Create hdf5
    hdf5_path = os.path.join(workspace, "features", "cmplx_spectrogram.h5")
    create_folder(os.path.dirname(hdf5_path))
    
    with h5py.File(hdf5_path, 'w') as hf:
        
        hf.attrs['sample_rate'] = sample_rate
        hf.attrs['fft_size'] = fft_size
        hf.attrs['hop_size'] = hop_size
        hf.attrs['window_type'] = window_type
    
    # Write out features to hdf5
    write_features_to_hdf5(tr_speech_dir, hdf5_path, 'train', 'speech', sample_rate, fft_size, hop_size, window_type)
    write_features_to_hdf5(tr_noise_dir, hdf5_path, 'train', 'noise', sample_rate, fft_size, hop_size, window_type)
    write_features_to_hdf5(te_speech_dir, hdf5_path, 'test', 'speech', sample_rate, fft_size, hop_size, window_type)
    write_features_to_hdf5(te_noise_dir, hdf5_path, 'test', 'noise', sample_rate, fft_size, hop_size, window_type)
    
    print("Write out to hdf5_path: %s" % hdf5_path)

    

def write_features_to_hdf5(audio_dir, hdf5_path, data_type, audio_type, sample_rate, fft_size, hop_size, window_type):
    
    n_freq = fft_size // 2 + 1
    
    if window_type == 'hamming':
        window = np.hamming(fft_size)
        
    print("--- %s, %s ---" % (data_type, audio_type))
    
    # Create group
    with h5py.File(hdf5_path, 'a') as hf:
        
        if data_type not in hf.keys():
            hf.create_group(data_type)
            
        if audio_type not in hf[data_type].keys():
            hf[data_type].create_group(audio_type)
        
        hf[data_type][audio_type].create_dataset(
            name='data', 
            shape=(0, n_freq), 
            maxshape=(None, n_freq), 
            dtype=np.complex64)
            
        hf[data_type][audio_type].create_dataset(
            name='raw', 
            shape=(0, fft_size), 
            maxshape=(None, fft_size), 
            dtype=np.float32)
        
        hf_data = hf[data_type][audio_type]['data']
        hf_raw = hf[data_type][audio_type]['raw']
        
        name_list = []
        bgn_fin_indices  = []
        energy_list = []
        
        # Spectrogram of a song
        names = os.listdir(audio_dir)
        
        for na in names:
            
            # Extract spectrogram & raw audio frames
            audio_path = os.path.join(audio_dir, na)
            (audio, _) = read_audio(audio_path, sample_rate)
            
            audio = normalize(audio)
            sp = calc_sp(audio, fft_size, hop_size, window)
            frames = stft.enframe(audio, fft_size, hop_size)
            energy = calculate_energy(audio)
            
            print(audio_path, sp.shape, "eng:", energy)
            
            # Write spectrogram & raw audio frames out to hdf5
            bgn_indice = hf_data.shape[0]
            fin_indice = bgn_indice + sp.shape[0]
            
            hf_data.resize((fin_indice, n_freq))
            hf_raw.resize((fin_indice, fft_size))
            
            hf_data[bgn_indice : fin_indice] = sp
            hf_raw[bgn_indice : fin_indice] = frames

            name_list.append(na)
            energy_list.append(energy)
            bgn_fin_indices.append((bgn_indice, fin_indice))
            
        # Write out bin_fin_indices to hdf5
        hf[data_type][audio_type].create_dataset(
            name='bgn_fin_indices', 
            data=bgn_fin_indices, 
            dtype=np.int32)
            
        # Write out energy_list
        hf[data_type][audio_type].create_dataset(
            name='energies', 
            data=energy_list, 
            dtype=np.float32)
            
        # Write out name_list to hdf5
        hf[data_type][audio_type].create_dataset(
            name='names', 
            data=name_list, 
            dtype='S64')







if __name__ == '__main__':
    calculate_features()