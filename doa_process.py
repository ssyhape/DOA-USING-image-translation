import numpy as np
from scipy.io import loadmat
import scipy.signal as signal
import matplotlib.pyplot as plt
import torch
import os
import cv2
import random
import math
os.environ['KMP_DUPLICATE_LIB_OK']='True'
datasource = loadmat('./signal_set.mat')['signal_list']
angle_set = loadmat('./angle_set.mat')['angle_list'] # T * N
dis_set = loadmat('./dis_set.mat')['dis_list'] # T *  N
loc_set = loadmat('./loc_set.mat')['location_list'] # T * 3  3 = dim
datasource = np.array([datasource[i:i+12] for i in range(0, len(datasource), 12)])
datasource = np.array([datasource[:,i:i+4,:] for i in range(0,datasource.shape[1],4)])#N * T * K * L

degree = np.arange(-90,91,1)
radians = np.deg2rad(degree)


range_dis = np.arange(0,11,0.02)

def stft_process(F_length,processed_origin):
    """
    processed_origin:  T * K *L ,T =600
    F_length : the window length of FFT  = 63 * 2
    L_length : the num of snapshots
    S = l/ F
    """
    T = processed_origin.shape[0]
    data_after_stft =[]
    f_list = []
    for t in range(T):
        signal_data = processed_origin[t]
        stft_results = []
        for i in range(signal_data.shape[0]):
            f,t,Zxx = signal.stft(signal_data[i],fs=16000,window='hamming',nperseg=F_length,noverlap=0, return_onesided=True)
            stft_results.append(Zxx)

            f_list.append(f)

        stft_results = np.array(stft_results) # k*F*S the aim is TS*F*K
        stft_results = np.transpose(stft_results,(2,1,0))
        data_after_stft.append(stft_results)
    data_after_stft = np.concatenate(data_after_stft,axis=0)
    f_list = f_list[0]
    return data_after_stft,f_list

def doa_range_feature_extraction(processed_origin_per_frame,theta_operator,range_operator):
    """
    processed_origin_per_frame : data F*k
    theta_operator: w_theta S*k
    range_operator: w_range F*D
    """
    return np.abs((theta_operator.dot(processed_origin_per_frame.T)).dot(range_operator))

def  compensation_range_method(frame,range_hat,range_real):
    """
    range_hat :tdis predicted
    """
    frame_a = np.copy(frame)
    for k in range(frame.shape[1]):
        cc = c
        tt = np.exp(1j * 2 * np.pi * f_list * (range_hat - range_real) / (c))
        frame_a[:,k] =frame [:,k] * np.exp(1j*2*np.pi*f_list*(range_hat- range_real)/(c))

    return frame_a

def search_range_predict(doa_feature):
    """
    return back is the hat range of doa_feature
    """
    max_index = np.argmax(doa_feature)
    return 4*np.unravel_index(max_index,doa_feature.shape)[1]/len(range_dis)

def normolaize_01(a):
    return (a-a.min())/(a.max()-a.min())

def location_target_plot(tmp_plot,theta,x_ind,y_ind):
    """
    tmp_plot is the semi work
    theta : the hyperparameter controlling the rate of decay
    """

    for i in range(400):
        for j in range(250):
            tmp_plot[i][j] = np.exp(-((i-x_ind)**2+(j-y_ind)**2)/(theta**2))
    return tmp_plot

def doa_2_xy2(channel_num,doa_per_frame):
    micro_xy = [[[0,2.461],[0,2.487],[0,2.513],[0,2.539]],
                [[3.961,0],[3.987,0],[4.013,0],[4.039,0]],
                [[8,2.461],[8,2.487],[8,2.513],[8,2.539]]]
    micro_xy = np.array(micro_xy)
    ap_pos = micro_xy[channel_num]
    d1 = np.array(list(range(int(8/0.02))))[np.newaxis,:].T
    d2 = np.array(list(range(int(5/0.02))))[np.newaxis,:].T

    P_out = np.zeros((len(d2),len(d1)))
    ap_center = np.mean(ap_pos,axis=0)
    X = np.tile(d1.T,(len(d2),1)) - ap_center[0]
    Y = np.tile(d2,(1,len(d1))) - ap_center[1]

    ap_vec = (ap_pos[0] - ap_pos[-1])

    T_n = []
    for x,y in zip(np.nditer(X,order = 'F'),np.nditer(Y,order ='F')):
        T_n.append(np.sign(np.sum([x,y]*ap_vec)) * (np.pi/2 - np.arccos(np.abs(np.sum([x,y]*ap_vec)))) / np.linalg.norm([x,y])/ np.linalg.norm(ap_vec))
    T_n = np.array(T_n)
    D = np.sqrt(X**2 + Y**2)
    D = D.flatten(order = 'F')
    T_IDX = np.argmin(np.abs(np.tile(T_n[:,np.newaxis],(1,len(radians))) - np.tile(radians.T,(len(T_n),1))) , axis=1)
    D_IDX = np.argmin(np.abs(np.tile(D[:,np.newaxis],(1,len(range_dis))) - np.tile(range_dis.T,(len(D),1))), axis=1)
    
    IDX = np.ravel_multi_index((T_IDX,D_IDX),dims = doa_per_frame.shape)
    IDX = np.reshape(IDX,(len(d2),len(d1)),order = 'F')
    p = doa_per_frame.flatten(order='F')
    P_out =p[IDX]

    return P_out

def doa_2_xy(channel_num,doa_per_frame):
    ref_xy = [[0,2.5],[4,0],[8,2.5]]
    XY_per_frame = np.zeros((int(8/0.02),int(5/0.02)))
    if channel_num ==0:
        for i in range(doa_per_frame.shape[0]):
            angle = i-90 # 对应角度
            angle_rad = math.radians(angle)

            dx = range_dis * math.cos(angle_rad)
            dy = range_dis* math.sin(angle_rad)
            x = ref_xy[channel_num][0] + dx
            y = ref_xy[channel_num][1] + dy
            x = np.array(x // 0.02).astype(int)
            y = np.array(y // 0.02).astype(int)
            for d in range(doa_per_frame.shape[1]):
                try:
                    XY_per_frame[x[d]][y[d]] = doa_per_frame[i][d]
                except:
                    continue
    if channel_num ==1:
        for i in range(doa_per_frame.shape[0]):
            angle = i-90 # 对应角度
            angle_rad = math.radians(angle)

            dx = range_dis * math.cos(angle_rad)
            dy = range_dis * math.sin(angle_rad)
            x = ref_xy[channel_num][0] + dy
            y = ref_xy[channel_num][1] + dx
            x = np.array(x // 0.02).astype(int)
            y = np.array(y // 0.02).astype(int)
            for d in range(doa_per_frame.shape[1]):
                try:
                    XY_per_frame[x[d]][y[d]] = doa_per_frame[i][d]
                except:
                    continue

    if channel_num ==2:
        for i in range(doa_per_frame.shape[0]):
            angle = i-90 # 对应角度
            angle_rad = math.radians(angle)

            dx = range_dis * math.cos(angle_rad)
            dy = range_dis * math.sin(angle_rad)
            x = ref_xy[channel_num][0] - dx
            y = ref_xy[channel_num][1] + dy
            x = np.array(x // 0.02).astype(int)
            y = np.array(y // 0.02).astype(int)
            for d in range(doa_per_frame.shape[1]):
                try:
                    XY_per_frame[x[d]][y[d]] = doa_per_frame[i][d]
                except:
                    continue
    return XY_per_frame

#Process datasource info
dis_u = 0.026
c = 340
channel_num = datasource.shape[0]
data_after_stft = []
for channel in range(channel_num):
    signal_tmp,f_list = stft_process(126,datasource[channel])
    data_after_stft.append(signal_tmp)
data_after_stft = np.array(data_after_stft) # N * TS *F *K
median_f = np.median(f_list)

# Process target info
angle_set = np.repeat(angle_set,18,axis =0 ) # TS * N
dis_set = np.repeat(dis_set,18,axis =0)#TS * N
loc_set = np.repeat(loc_set,18,axis =0)
loc_set = loc_set[:,0:2]
#DOA extract operator
processed_origin_per_frame = data_after_stft[0][0]
theta_operator = []
for k in range(1,processed_origin_per_frame.shape[1]+1):
    theta_operator.append(np.exp(-1j * (2 * np.pi *dis_u *np.sin(radians)*k*median_f/c)))
theta_operator = np.conjugate(np.array(theta_operator))
theta_operator = np.transpose(theta_operator)

range_operator = []
for f in range(1, processed_origin_per_frame.shape[0] + 1):
    range_operator.append(np.exp(-1j * (2 * np.pi * f_list[f-1] * f * range_dis /( c))))
range_operator = np.conjugate(np.array(range_operator))


#order set


#save doa_feature
for i in range(108):
    doa_extracted_ = []
    for channel in range(3):
        tmp_per_c = []
        for frame in data_after_stft[channel][0+i*100:100+i*100]:
            tt1 = doa_2_xy(channel,doa_range_feature_extraction(frame, theta_operator, range_operator))
            tt1 = cv2.blur(tt1,(2,2))
            tmp_per_c.append(doa_2_xy(channel,doa_range_feature_extraction(frame, theta_operator, range_operator)))
        doa_extracted_.append(tmp_per_c)
    doa_extracted_ = torch.tensor(doa_extracted_) # N * 100 * X * Y
    filename = f"doa_train\doa_extract_{i}.pt"
    torch.save(doa_extracted_,filename)

del doa_extracted_



#save composentation doa_f
for i in range(108):
    doa_extracted_ = []
    for channel in range(3):
        tmp = []
        for j,frame in enumerate(data_after_stft[channel][0+i*100:100+i*100]):
            range_hat = search_range_predict(doa_range_feature_extraction(frame, theta_operator, range_operator))
            range_real = dis_set[j+i*100][channel]
            c_t = compensation_range_method(frame, range_hat, range_real)
            tt2 = doa_2_xy(channel,doa_range_feature_extraction( c_t, theta_operator, range_operator))
            tt2 = cv2.blur(tt2,[2,2])
            tmp.append(tt2)
        doa_extracted_.append(tmp)
    doa_extracted_ = torch.tensor(doa_extracted_)
    filename = f"./target1/doa_extract_compensation_{i}.pt"
    torch.save(doa_extracted_,filename)

#save target 2
for i in range(108):
    target2 = []
    for index in loc_set[0+i*100:100+i*100,:]:
        tmp_plot = np.zeros((400,250))
        tmp_plot = location_target_plot(tmp_plot,7.5,int(index[0]//0.02),int(index[1]//0.02))
        target2.append(tmp_plot)
    target2 = torch.tensor(target2)
    filename = f'./target2/target2_{i}.pt'
    torch.save(target2,filename)


"""

a =doa_range_feature_extraction(data_after_stft[0][110],theta_operator,range_operator)
b = doa_2_xy2(0,a)
"""
