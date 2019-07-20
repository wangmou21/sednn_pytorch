# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:54:48 2019

@author: Mou
"""

import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data_generator import DataGenerator
from spectrogram_to_wave import recover_wav
from utils import (load_hdf5, scale_on_3d, scale_on_2d, create_folder, write_audio,
    np_mean_absolute_error, pad_with_border, log_sp, mat_2d_to_3d, inverse_scale_on_2d)
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class sednn(nn.Module):
    def __init__(self, n_concat, n_freq, n_hid):
        super(sednn, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_concat*n_freq, n_hid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hid, n_freq)
            )
        
    def forward(self, x):
        x = x.reshape(x.size(0),-1)
        x = self.net(x)
        return x


def train():
    """Train the neural network. Write out model every several iterations. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      lr: float, learning rate. 
    """

    workspace = config.workspace
    tr_snr = config.Tr_SNR
    te_snr = config.Te_SNR
    lr = config.lr
    n_hid = config.n_hid
    cuda = config.cuda
    
    # Load data. 
    t1 = time.time()
    tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "data.h5")
    te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "%ddb" % int(te_snr), "data.h5")
    (tr_x, tr_y) = load_hdf5(tr_hdf5_path)
    (te_x, te_y) = load_hdf5(te_hdf5_path)
    print(tr_x.shape, tr_y.shape)
    print(te_x.shape, te_y.shape)
    print("Load data time: %s s" % (time.time() - t1,))
    
    batch_size = 500
    print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))
    
    # Scale data. 
    if True:
        t1 = time.time()
        scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "scaler.p")
        scaler = pickle.load(open(scaler_path, 'rb'))
        tr_x = scale_on_3d(tr_x, scaler)
        tr_y = scale_on_2d(tr_y, scaler)
        te_x = scale_on_3d(te_x, scaler)
        te_y = scale_on_2d(te_y, scaler)
        print("Scale data time: %s s" % (time.time() - t1,))
        
    # Debug plot. 
    if False:
        plt.matshow(tr_x[0 : 1000, 0, :].T, origin='lower', aspect='auto', cmap='jet')
        plt.show()
        pause
        
    # Build model
    (_, n_concat, n_freq) = tr_x.shape
    
    model = sednn(n_concat, n_freq, n_hid)
    if cuda:
        model.cuda()
        
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)
    
    criterion = nn.MSELoss()
    
    # Data generator. 
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    
    # Directories for saving models and training stats
    model_dir = os.path.join(workspace, "models", "%ddb" % int(tr_snr))
    create_folder(model_dir)
    
    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    create_folder(stats_dir)
    
    # Print loss before training. 
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
    te_loss = eval(model, eval_te_gen, te_x, te_y)
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
    
    # Save out training stats. 
    stat_dict = {'iter': iter, 
                    'tr_loss': tr_loss, 
                    'te_loss': te_loss, }
    stat_path = os.path.join(stats_dir, "%diters.p" % iter)
    pickle.dump(stat_dict, open(stat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    # Train. 
    t1 = time.time()

    
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        batch_x = Variable(torch.Tensor(batch_x), requires_grad = False).cuda() if cuda else Variable(batch_x)
        batch_y = Variable(torch.Tensor(batch_y), requires_grad = False).cuda() if cuda else Variable(batch_y)
        
        batch_y_pre = model(batch_x)
        loss = criterion(batch_y_pre, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        iter += 1
        
        # Validate and save training stats. 
        if iter % 1000 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
            te_loss = eval(model, eval_te_gen, te_x, te_y)
            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
            
            # Save out training stats. 
            stat_dict = {'iter': iter, 
                         'tr_loss': tr_loss, 
                         'te_loss': te_loss, }
            stat_path = os.path.join(stats_dir, "%diters.p" % iter)
            pickle.dump(stat_dict, open(stat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            
        # Save model. 
        if iter % 5000 == 0:
            
            checkpoint = {
                'iteration': iter, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}
                        
                        
            checkpoint_path = os.path.join(model_dir, "md_%diters.h5" % iter)
                
            torch.save(checkpoint, checkpoint_path)
            print("Saved model to %s" % checkpoint_path)
        
        if iter == 10001:
            break
            
    print("Training time: %s s" % (time.time() - t1,))

def eval(model, gen, x, y):
    """Validation function. 
    
    Args:
      model: keras model. 
      gen: object, data generator. 
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    cuda = config.cuda
    pred_all, y_all = [], []
    
    # Inference in mini batch. 
    for (batch_x, batch_y) in gen.generate(xs=[x], ys=[y]):
        batch_x = Variable(torch.Tensor(batch_x), requires_grad = False).cuda() if cuda else Variable(batch_x)
        pred = model(batch_x)
        pred_all.append(pred.detach().cpu().numpy())
        y_all.append(batch_y)
        
    # Concatenate mini batch prediction. 
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    # Compute loss. 
    loss = np_mean_absolute_error(y_all, pred_all)
    return loss

def inference():
    """Inference all test data, write out recovered wavs to disk. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      n_concat: int, number of frames to concatenta, should equal to n_concat 
          in the training stage. 
      iter: int, iteration of model to load. 
      visualize: bool, plot enhanced spectrogram for debug. 
    """
#    print(args)
    workspace = config.workspace
    tr_snr = config.Tr_SNR
    te_snr = config.Te_SNR
    n_concat = config.n_concat
    iter = config.iteration
    cuda = config.cuda
    visualize = config.visualize
    
    n_window = config.n_window
    n_overlap = config.n_overlap
    fs = config.sample_rate
    n_hid = config.n_hid
    scale = True
    n_freq = int(config.fft_size/2+1)
    
    # Load model. 
   
    model = sednn(n_concat, n_freq, n_hid)
    if cuda:
        model.cuda()
    checkpoint_path = os.path.join(workspace, "models", "%ddb" % int(tr_snr), "md_%diters.h5" % iter)   
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    # Load scaler. 
    scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    # Load test data. 
    feat_dir = os.path.join(workspace, "features", "spectrogram", "test", "%ddb" % int(te_snr))
    names = os.listdir(feat_dir)

    for (cnt, na) in enumerate(names):
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = pickle.load(open(feat_path, 'rb'))
        [mixed_cmplx_x, speech_x, noise_x, alpha, na] = data
        mixed_x = np.abs(mixed_cmplx_x)
        
        # Process data. 
        n_pad = int((n_concat - 1) / 2)
        mixed_x = pad_with_border(mixed_x, n_pad)
        mixed_x = log_sp(mixed_x)
        speech_x = log_sp(speech_x)
        
        # Scale data. 
        if scale:
            mixed_x = scale_on_2d(mixed_x, scaler)
            speech_x = scale_on_2d(speech_x, scaler)
        
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        
        # Predict. 
        mixed_x_3d = Variable(torch.Tensor(mixed_x_3d), requires_grad = False).cuda()
        pred = model(mixed_x_3d)
        pred = pred.detach().cpu().numpy()
        print(cnt, na)
        
        # Inverse scale. 
        if scale:
            mixed_x = inverse_scale_on_2d(mixed_x, scaler)
            speech_x = inverse_scale_on_2d(speech_x, scaler)
            pred = inverse_scale_on_2d(pred, scaler)
        
        # Debug plot. 
        if visualize:
            fig, axs = plt.subplots(3,1, sharex=False)
            axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
            axs[0].set_title("%ddb mixture log spectrogram" % int(te_snr))
            axs[1].set_title("Clean speech log spectrogram")
            axs[2].set_title("Enhanced speech log spectrogram")
            for j1 in range(3):
                axs[j1].xaxis.tick_bottom()
            plt.tight_layout()
            plt.show()

        # Recover enhanced wav. 
        pred_sp = np.exp(pred)
        s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
        s *= np.sqrt((np.hamming(n_window)**2).sum())   # Scaler for compensate the amplitude 
                                                        # change after spectrogram and IFFT. 
        
        # Write out enhanced wav. 
        out_path = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr), "%s.enh.wav" % na)
        create_folder(os.path.dirname(out_path))
        write_audio(out_path, s, fs)

if __name__ == '__main__':
    
    train()   
    # Inference, enhanced wavs will be created. 
    inference()
