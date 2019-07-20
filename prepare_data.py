# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:39:18 2019

@author: Mou
"""

import os
import csv
import time
import numpy as np
import pickle
import h5py
from scipy import signal
from sklearn.preprocessing import StandardScaler

from utils import create_folder, read_audio, write_audio, pad_with_border, log_sp, mat_2d_to_3d
import config



def create_mixture_csv(data_type):
    """Create csv containing mixture information. 
    Each line in the .csv file contains [speech_name, noise_name, noise_onset, noise_offset]
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      magnification: int, only used when data_type='train', number of noise 
          selected to mix with a speech. E.g., when magnication=3, then 4620
          speech with create 4620*3 mixtures. magnification should not larger 
          than the species of noises. 
    """
    
    workspace = config.workspace
    data_dir = config.data_dir
    speech_dir = os.path.join(data_dir,'{}_speech'.format(data_type))
    noise_dir = os.path.join(data_dir,'{}_noise'.format(data_type))   
    magnification = config.magnification
    fs = config.sample_rate
    
    speech_names = [na for na in os.listdir(speech_dir) if na.lower().endswith(".wav")]
    noise_names = [na for na in os.listdir(noise_dir) if na.lower().endswith(".wav")]
    
    rs = np.random.RandomState(0)
    out_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
    create_folder(os.path.dirname(out_csv_path))
    
    cnt = 0
    f = open(out_csv_path, 'w')
    f.write("%s\t%s\t%s\t%s\n" % ("speech_name", "noise_name", "noise_onset", "noise_offset"))
    for speech_na in speech_names:
        # Read speech. 
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path)
        len_speech = len(speech_audio)
        
        # For training data, mix each speech with randomly picked #magnification noises. 
        if data_type == 'train':
            selected_noise_names = rs.choice(noise_names, size=magnification, replace=False)
        # For test data, mix each speech with all noises. 
        elif data_type == 'test':
            selected_noise_names = noise_names
        else:
            raise Exception("data_type must be train | test!")

        # Mix one speech with different noises many times. 
        for noise_na in selected_noise_names:
            noise_path = os.path.join(noise_dir, noise_na)
            (noise_audio, _) = read_audio(noise_path)
            
            len_noise = len(noise_audio)

            if len_noise <= len_speech:
                noise_onset = 0
                nosie_offset = len_speech
            # If noise longer than speech then randomly select a segment of noise. 
            else:
                noise_onset = rs.randint(0, len_noise - len_speech, size=1)[0]
                nosie_offset = noise_onset + len_speech
            
            if cnt % 100 == 0:
                print(cnt)
                
            cnt += 1
            f.write("%s\t%s\t%d\t%d\n" % (speech_na, noise_na, noise_onset, nosie_offset))
    f.close()
    print(out_csv_path)
    print("Create %s mixture csv finished!" % data_type)


def calculate_mixture_features(data_type):
    """Calculate spectrogram for mixed, speech and noise audio. Then write the 
    features to disk. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
    """
    workspace = config.workspace
    data_dir = config.data_dir
    speech_dir = os.path.join(data_dir,'{}_speech'.format(data_type))
    noise_dir = os.path.join(data_dir,'{}_noise'.format(data_type))   
    
    fs = config.sample_rate
    
    if data_type == 'train':
        snr = config.Tr_SNR
    elif data_type == 'test':
        snr = config.Te_SNR    
    else:
        raise Exception("data_type must be train | test!")
    
    
    # Open mixture csv. 
    mixture_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
    with open(mixture_csv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
    
    t1 = time.time()
    cnt = 0
    for i1 in range(1, len(lis)):
        [speech_na, noise_na, noise_onset, noise_offset] = lis[i1]
        noise_onset = int(noise_onset)
        noise_offset = int(noise_offset)
        
        # Read speech audio. 
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path, target_fs=fs)
        
        # Read noise audio. 
        noise_path = os.path.join(noise_dir, noise_na)
        (noise_audio, _) = read_audio(noise_path, target_fs=fs)
        
        # Repeat noise to the same length as speech. 
        if len(noise_audio) < len(speech_audio):
            n_repeat = int(np.ceil(float(len(speech_audio)) / float(len(noise_audio))))
            noise_audio_ex = np.tile(noise_audio, n_repeat)
            noise_audio = noise_audio_ex[0 : len(speech_audio)]
        # Truncate noise to the same length as speech. 
        else:
            noise_audio = noise_audio[noise_onset : noise_offset]
        
        # Scale speech to given snr. 
        scaler = get_amplitude_scaling_factor(speech_audio, noise_audio, snr=snr)
        speech_audio *= scaler
        
        # Get normalized mixture, speech, noise. 
        (mixed_audio, speech_audio, noise_audio, alpha) = additive_mixing(speech_audio, noise_audio)

        # Write out mixed audio. 
        out_bare_na = os.path.join("%s.%s" % 
            (os.path.splitext(speech_na)[0], os.path.splitext(noise_na)[0]))
        out_audio_path = os.path.join(workspace, "mixed_audios", "spectrogram", 
            data_type, "%ddb" % int(snr), "%s.wav" % out_bare_na)
        create_folder(os.path.dirname(out_audio_path))
        write_audio(out_audio_path, mixed_audio, fs)

        # Extract spectrogram. 
        mixed_complx_x = calc_sp(mixed_audio, mode='complex')
        speech_x = calc_sp(speech_audio, mode='magnitude')
        noise_x = calc_sp(noise_audio, mode='magnitude')

        # Write out features. 
        out_feat_path = os.path.join(workspace, "features", "spectrogram", 
            data_type, "%ddb" % int(snr), "%s.p" % out_bare_na)
        create_folder(os.path.dirname(out_feat_path))
        data = [mixed_complx_x, speech_x, noise_x, alpha, out_bare_na]
        pickle.dump(data, open(out_feat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        # Print. 
        if cnt % 100 == 0:
            print(cnt)
            
        cnt += 1

    print("Extracting feature time: %s" % (time.time() - t1))

def rms(y):
    """Root mean square. 
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))

def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    """Given s and n, return the scaler s according to the snr. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      snr: float, SNR. 
      method: 'rms'. 
      
    Outputs:
      float, scaler. 
    """
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio =  10. ** (float(snr) / 20.)    # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor

def additive_mixing(s, n):
    """Mix normalized source1 and source2. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      
    Returns:
      mix_audio: ndarray, mixed audio. 
      s: ndarray, pad or truncated and scalered source1. 
      n: ndarray, scaled source2. 
      alpha: float, normalize coefficient. 
    """
    mixed_audio = s + n
        
    alpha = 1. / np.max(np.abs(mixed_audio))
    mixed_audio *= alpha
    s *= alpha
    n *= alpha
    return mixed_audio, s, n, alpha

def calc_sp(audio, mode):
    """Calculate spectrogram. 
    
    Args:
      audio: 1darray. 
      mode: string, 'magnitude' | 'complex'
    
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    n_window = config.n_window
    n_overlap = config.n_overlap
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode=mode) 
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x

def pack_features(data_type):
    """Load all features, apply log and conver to 3D tensor, write out to .h5 file. 
    
    Args:
      workspace: str, path of workspace. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
      n_concat: int, number of frames to be concatenated. 
      n_hop: int, hop frames. 
    """
    workspace = config.workspace

    if data_type == 'train':
        snr = config.Tr_SNR
    elif data_type == 'test':
        snr = config.Te_SNR    
    else:
        raise Exception("data_type must be train | test!")
        
    n_concat = config.n_concat
    n_hop = config.n_hop
    
    x_all = []  # (n_segs, n_concat, n_freq)
    y_all = []  # (n_segs, n_freq)
    
    cnt = 0
    t1 = time.time()
    
    # Load all features. 
    feat_dir = os.path.join(workspace, "features", "spectrogram", data_type, "%ddb" % int(snr))
    names = os.listdir(feat_dir)
    for na in names:
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = pickle.load(open(feat_path, 'rb'))
        [mixed_complx_x, speech_x, noise_x, alpha, na] = data
        mixed_x = np.abs(mixed_complx_x)

        # Pad start and finish of the spectrogram with boarder values. 
        n_pad = int((n_concat - 1) / 2)
        mixed_x = pad_with_border(mixed_x, n_pad)
        speech_x = pad_with_border(speech_x, n_pad)
    
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=n_hop)
        x_all.append(mixed_x_3d)
        
        # Cut target spectrogram and take the center frame of each 3D segment. 
        speech_x_3d = mat_2d_to_3d(speech_x, agg_num=n_concat, hop=n_hop)
        y = speech_x_3d[:, int((n_concat-1)/2), :]
        y_all.append(y)
    
        # Print. 
        if cnt % 100 == 0:
            print(cnt)
            
        # if cnt == 3: break
        cnt += 1
        
    x_all = np.concatenate(x_all, axis=0)   # (n_segs, n_concat, n_freq)
    y_all = np.concatenate(y_all, axis=0)   # (n_segs, n_freq)
    
    x_all = log_sp(x_all).astype(np.float32)
    y_all = log_sp(y_all).astype(np.float32)
    
    # Write out data to .h5 file. 
    out_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    create_folder(os.path.dirname(out_path))
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
    
    print("Write out to %s" % out_path)
    print("Pack features finished! %s s" % (time.time() - t1,))

def compute_scaler(data_type):
    """Compute and write out scaler of data. 
    """
    workspace = config.workspace

    if data_type == 'train':
        snr = config.Tr_SNR
    
    # Load data. 
    t1 = time.time()
    hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')     
        x = np.array(x)     # (n_segs, n_concat, n_freq)
    
    # Compute scaler. 
    (n_segs, n_concat, n_freq) = x.shape
    x2d = x.reshape((n_segs * n_concat, n_freq))
    scaler = StandardScaler(with_mean=True, with_std=True).fit(x2d)
#    print(scaler.mean_)
#    print(scaler.scale_)
    
    # Write out scaler. 
    out_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "scaler.p")
    create_folder(os.path.dirname(out_path))
    pickle.dump(scaler, open(out_path, 'wb'))
    
    print("Save scaler to %s" % out_path)
    print("Compute scaler finished! %s s" % (time.time() - t1,))

if __name__ == '__main__':
 
    ## train 
    
    data_type = 'train'  
    # Create mixture csv. 
    create_mixture_csv(data_type)
    # Calculate mixture features.
    calculate_mixture_features(data_type)
    
    pack_features(data_type) 
    
    compute_scaler(data_type)
    
    ## test
    
    data_type = 'test'  
    # Create mixture csv. 
    create_mixture_csv(data_type)
    # Calculate mixture features.
    calculate_mixture_features(data_type)
    
    pack_features(data_type) 
