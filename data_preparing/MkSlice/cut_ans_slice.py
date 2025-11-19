# 現状未使用


import numpy as np
#import pyinterp
import copy
from scipy.ndimage import rotate  #conda
#
def sgcr_cut(ans_img2, v, SLB, SLB2):
    lnum = np.unique(ans_img2[v])[1:]
    cnt = len(lnum)
    if cnt<=0:
        cnt=1
    tmp15_1=[]
    tmp15_2=[]
    for h in range(cnt):###ラベルの数
        # Use boolean indexing instead of deepcopy for better memory efficiency
        tmp = np.zeros_like(ans_img2[v])
        if len(lnum) > 0:
            tmp[ans_img2[v] == lnum[h]] = 1
        tmp2 = rotate(tmp, angle=45, axes=(0, 1), reshape=True)
        tmp2[tmp2 < 0.5] = 0
        tmp2[tmp2 > 0] = 1
        tmp15_3=[]
        tmp15_4=[]
        #######切り出し１
        kkk = len(tmp2[:,0,0])
        sx = len(tmp2[0,:,0])
        sy = len(tmp2[0,0,:])
        for i in range(SLB2,kkk-SLB2):###スライスの数
            ######スラブ１５
            tmp3 = tmp2[i,SLB2:(sx - SLB2), SLB:(sy - SLB)]
            sum = tmp3
            for k in range(-7, 7):
                if (k == 0):
                    continue
                tmp3 =  tmp2[i+k,SLB2:(sx - SLB2), SLB:(sy - SLB)]
                sum = sum + tmp3
            tmp15_3.append(sum.copy())
        print("tmp15_3", tmp15_3[0].shape)
        print("len(tmp15_3)",len(tmp15_3))
        tmp15_1.append(tmp15_3)
        #######切り出し2
        kkk = len(tmp2[0, :, 0])
        sx = len(tmp2[:, 0, 0])
        sy = len(tmp2[0, 0, :])
        for i in range(SLB2, kkk - SLB2):  ###スライスの数
            ######スラブ１５
            tmp3 = tmp2[SLB2:(sx - SLB2), i, SLB:(sy - SLB)]
            sum = tmp3
            for k in range(-7, 7):
                if (k == 0):
                    continue
                tmp3 = tmp2[ SLB2:(sx - SLB2), i + k, SLB:(sy - SLB)]
                sum = sum + tmp3
            tmp15_4.append(sum.copy())
        print("tmp15_4", tmp15_4[0].shape)
        print("len(tmp15_4)", len(tmp15_4))
        tmp15_2.append(tmp15_4)
    print("len(tmp15_1)", len(tmp15_1))
    print("len(tmp15_2)", len(tmp15_2))

    return tmp15_1, tmp15_2


def crax_cut(ans_img2, v, SLB, SLB2):
    lnum = np.unique(ans_img2[v])[1:]
    cnt = len(lnum)
    if cnt<=0:
        cnt=1
    tmp15_1=[]
    tmp15_2=[]
    for h in range(cnt):###ラベルの数
        # Use boolean indexing instead of deepcopy for better memory efficiency
        tmp = np.zeros_like(ans_img2[v])
        if len(lnum) > 0:
            tmp[ans_img2[v] == lnum[h]] = 1
        tmp2 = rotate(tmp, angle=45, axes=(1, 2), reshape=True)
        tmp2[tmp2 < 0.5] = 0
        tmp2[tmp2 > 0] = 1
        tmp15_3=[]
        tmp15_4=[]
        #######切り出し１
        kkk = len(tmp2[0,:,0])
        sx = len(tmp2[:,0,0])
        sy = len(tmp2[0,0,:])
        for i in range(SLB2,kkk-SLB2):###スライスの数
            ######スラブ１５
            tmp3 = tmp2[SLB:(sx - SLB), i, SLB2:(sy - SLB2)]
            sum = tmp3
            for k in range(-7, 7):
                if (k == 0):
                    continue
                tmp3 =  tmp2[SLB:(sx - SLB), i+k, SLB2:(sy - SLB2)]
                sum = sum + tmp3
            tmp15_3.append(np.array(sum).T.copy())
        print("tmp15_3", tmp15_3[0].shape)
        print("len(tmp15_3)",len(tmp15_3))
        tmp15_1.append(tmp15_3)
        #######切り出し2
        kkk = len(tmp2[0, 0, :])
        sx = len(tmp2[:, 0, 0])
        sy = len(tmp2[0, :, 0])
        for i in range(SLB2, kkk - SLB2):  ###スライスの数
            ######スラブ１５
            tmp3 = tmp2[ SLB:(sx - SLB), SLB2:(sy - SLB2), i]
            sum = tmp3
            for k in range(-7, 7):
                if (k == 0):
                    continue
                tmp3 = tmp2[ SLB:(sx - SLB), SLB2:(sy - SLB2),  i + k]
                sum = sum + tmp3
            tmp15_4.append(np.array(sum).T.copy())
        print("tmp15_4", tmp15_4[0].shape)
        print("len(tmp15_4)", len(tmp15_4))
        tmp15_2.append(tmp15_4)
    print("len(tmp15_1)", len(tmp15_1))
    print("len(tmp15_2)", len(tmp15_2))

    return tmp15_1, tmp15_2


def axsg_cut(ans_img2, v, SLB, SLB2):
    lnum = np.unique(ans_img2[v])[1:]
    cnt = len(lnum)
    if cnt<=0:
        cnt=1
    tmp15_1=[]
    tmp15_2=[]
    for h in range(cnt):###ラベルの数
        # Use boolean indexing instead of deepcopy for better memory efficiency
        tmp = np.zeros_like(ans_img2[v])
        if len(lnum) > 0:
            tmp[ans_img2[v] == lnum[h]] = 1
        tmp2 = rotate(tmp, angle=45, axes=(0, 2), reshape=True)
        tmp2[tmp2 < 0.5] = 0
        tmp2[tmp2 > 0] = 1
        tmp15_3=[]
        tmp15_4=[]
        #######切り出し１
        kkk = len(tmp2[:,0,0])
        sx = len(tmp2[0,:,0])
        sy = len(tmp2[0,0,:])
        for i in range(SLB2,kkk-SLB2):###スライスの数
            ######スラブ１５
            tmp3 = tmp2[ i, SLB:(sx - SLB), SLB2:(sy - SLB2)]
            sum = tmp3
            for k in range(-7, 7):
                if (k == 0):
                    continue
                tmp3 =  tmp2[ i+k, SLB:(sx - SLB), SLB2:(sy - SLB2)]
                sum = sum + tmp3
            tmp15_3.append(np.array(sum).T.copy())
        print("tmp15_3", tmp15_3[0].shape)
        print("len(tmp15_3)",len(tmp15_3))
        tmp15_1.append(tmp15_3)
        #######切り出し2
        kkk = len(tmp2[0, 0, :])
        sx = len(tmp2[:, 0, 0])
        sy = len(tmp2[0, :, 0])
        for i in range(SLB2, kkk - SLB2):  ###スライスの数
            ######スラブ１５
            tmp3 = tmp2[ SLB2:(sx - SLB2), SLB:(sy - SLB), i]
            sum = tmp3
            for k in range(-7, 7):
                if (k == 0):
                    continue
                tmp3 = tmp2[ SLB2:(sx - SLB2), SLB:(sy - SLB),  i + k]
                sum = sum + tmp3
            tmp15_4.append(sum.copy())
        print("tmp15_4", tmp15_4[0].shape)
        print("len(tmp15_4)", len(tmp15_4))
        tmp15_2.append(tmp15_4)
    print("len(tmp15_1)", len(tmp15_1))
    print("len(tmp15_2)", len(tmp15_2))

    return tmp15_1, tmp15_2

def acs_cut(ans_img2, v, SLB):
    lnum = np.unique(ans_img2[v])[1:]
    cnt = len(lnum)
    if cnt<=0:
        cnt=1
    tmp15_1=[]
    tmp15_2=[]
    tmp15_3=[]
    for h in range(cnt):###ラベルの数
        # Use boolean indexing instead of deepcopy for better memory efficiency
        tmp2 = np.zeros_like(ans_img2[v])
        if len(lnum) > 0:
            tmp2[ans_img2[v] == lnum[h]] = 1
        #######切り出し１
        ccc=[]
        kkk = len(tmp2[:,0,0])
        sx = len(tmp2[0,:,0])
        sy = len(tmp2[0,0,:])
        for i in range(SLB,kkk-SLB):###スライスの数
            ######スラブ１５
            tmp3 = tmp2[i,SLB:(sx - SLB), SLB:(sy - SLB)]
            sum = tmp3
            for k in range(-7, 7):
                if (k == 0):
                    continue
                tmp3 =  tmp2[i+k,SLB:(sx - SLB), SLB:(sy - SLB)]
                sum = sum + tmp3
            ccc.append(sum.copy())
        print("ccc1", ccc[0].shape)
        print("len(ccc1)",len(ccc))
        tmp15_1.append(ccc)
        #######切り出し2
        ccc=[]
        kkk = len(tmp2[0, :, 0])
        sx = len(tmp2[:, 0, 0])
        sy = len(tmp2[0, 0, :])
        for i in range(SLB, kkk - SLB):  ###スライスの数
            ######スラブ１５
            tmp3 = tmp2[SLB:(sx - SLB), i, SLB:(sy - SLB)]
            sum = tmp3
            for k in range(-7, 7):
                if (k == 0):
                    continue
                tmp3 = tmp2[ SLB:(sx - SLB), i + k, SLB:(sy - SLB)]
                sum = sum + tmp3
            ccc.append(sum.copy())
        print("ccc2", ccc[0].shape)
        print("len(ccc2)", len(ccc))
        tmp15_2.append(ccc)
        #######切り出し3
        ccc=[]
        kkk = len(tmp2[0, 0, :])
        sx = len(tmp2[:, 0, 0])
        sy = len(tmp2[0, :, 0])
        for i in range(SLB, kkk - SLB):  ###スライスの数
            ######スラブ１５
            tmp3 = tmp2[SLB:(sx - SLB), SLB:(sy - SLB), i]
            sum = tmp3
            for k in range(-7, 7):
                if (k == 0):
                    continue
                tmp3 = tmp2[SLB:(sx - SLB), SLB:(sy - SLB), i + k]
                sum = sum + tmp3
            ccc.append(sum.copy())
        print("ccc3", ccc[0].shape)
        print("len(ccc3)", len(ccc))
        tmp15_3.append(ccc)
    print("len(tmp15_1)", len(tmp15_1))
    print("len(tmp15_2)", len(tmp15_2))
    print("len(tmp15_3)", len(tmp15_3))

    return tmp15_1, tmp15_2, tmp15_3
