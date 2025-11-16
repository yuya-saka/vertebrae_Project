from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageChops
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import subprocess
import math
#import cv2
#import glob
from apply_normalization import apply_normalization, gpu_rotate_3d
from cut_ans_slice import sgcr_cut, crax_cut, axsg_cut, acs_cut
import time
#from PIL import Image
#from scipy.ndimage.interpolation import rotate  #conda
from scipy.ndimage import rotate  #conda
import cc3d #connected-components-3d   pip
import copy
import gc
import os

####正規化後のサイズ
MAX_SIZE = 256
####椎骨切り出し範囲の拡張
MARGIN = 10 ###x,y方向の張量
SLB = 15 ##通常面の拡張量（スラブを作るために拡張し、スライス画像を保存するときには拡張前に戻る）
SLB2 = math.floor(SLB * math.sqrt(2))##クロス面の拡張量（スラブを作るために拡張し、スライス画像を保存するときには拡張前に戻る）
PLANE_NAME = ["Sagit","Coron","Axial","SgCr1","SgCr2","CrAx1","CrAx2","AxSg1","AxSg2"]

def mk_train_img(sbj_no,inp_nii,inp_path, seg_path, ans_path, an_opath, cn_opath, si_opath, si_opath_ans,si_opath_rect, al_opath, al_opath2, gpu_id=None):
    start_time = time.time()
    print(f"[INFO] Processing subject {sbj_no} - Start")

    inp_struct = nib.load(inp_path)
    seg_struct = nib.load(seg_path)
    ans_struct = nib.load(ans_path)

    inp = np.array(inp_struct.get_fdata())
    seg = np.array(seg_struct.get_fdata())
    ans = np.array(ans_struct.get_fdata())

    #print('[DEBUG:nii_load_info] Header : ', nii_struct.header)

    print('inp volume shape :', inp_struct.get_fdata().shape)
    print('inp resolution :', inp_struct.header.get_zooms())
    print('seg volume shape :', seg_struct.get_fdata().shape)
    print('seg resolution :', seg_struct.header.get_zooms())
    print('ans volume shape :', ans_struct.get_fdata().shape)
    print('ans resolution :', ans_struct.header.get_zooms())
    mx = inp_struct.get_fdata().shape[0]
    my = inp_struct.get_fdata().shape[1]
    mz = inp_struct.get_fdata().shape[2]
    print('mx :\t', mx)
    print('my :\t', my)
    print('mz :\t', mz)

    #image = Image.open("D:\TotalSeg\input\\test.png")
    #plt.imshow(image)
    #plt.show()

    # data_shape = math.floor(inp.shape[2]/2)
    # fig, ax = plt.subplots()
    # ax.imshow(inp[data_shape,:,:], cmap='gray', origin='lower')
    # ax.set_xlabel('z')
    # ax.set_ylabel('y')
    # plt.show()

    print('affine :',inp_struct.affine)
    print('header :',inp_struct.header)

    pix_spacing = [inp_struct.header.get_zooms()[0], inp_struct.header.get_zooms()[1]]
    slc_thickness = inp_struct.header.get_zooms()[2]

    print('pix_spacing :',pix_spacing)
    print('slc_thickness :',slc_thickness)

    #################################################################
    #椎骨外接検出(seg画像を使用)
    #################################################################
    ###########L5,L4,L3,L2,L1T12T11T10,T9,T8,T7,T6,T5,T4,
    vert_no = [27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    vert_no2 = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14]
    #vert_no = [31,32]
    max_x = [0]*len(vert_no)
    max_y = [0]*len(vert_no)
    max_z = [0]*len(vert_no)
    min_x = [mx]*len(vert_no)
    min_y = [my]*len(vert_no)
    min_z = [mz]*len(vert_no)
    for v in range(len(vert_no)):
        # vert_no[v] または vert_no2[v] に一致する座標を取得
        coords = np.argwhere((seg == vert_no[v]) | (seg == vert_no2[v]))
        if coords.size > 0:
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            min_x[v], min_y[v], min_z[v] = min_coords[0], min_coords[1], min_coords[2]
            max_x[v], max_y[v], max_z[v] = max_coords[0], max_coords[1], max_coords[2]

    #################################################################
    #椎骨切り出し範囲の計算(外接検出で求めた値を使用)
    #################################################################
    for v in range(len(vert_no)):
        if (max_x[v] == 0 or max_y[v] == 0 or max_z[v] == 0):
            continue
        max_x[v] = max_x[v] + MARGIN + SLB + 1#：の後ろはふくまれないため、１＋
        min_x[v] = min_x[v] - MARGIN - SLB
        max_y[v] = max_y[v] + MARGIN + SLB + 1#：の後ろはふくまれないため、１＋
        min_y[v] = min_y[v] - MARGIN - SLB
        max_z[v] = max_z[v] + SLB + 1#：の後ろはふくまれないため、１＋
        min_z[v] = min_z[v] - SLB
        ####腰椎から頭の方に向かって順に見ていく。
        ####切り出し範囲が、データ領域を超えるものが含まれる場合ブレークする。
        if (mx < max_x[v] or 0 > min_x[v] or my < max_y[v] or 0 > min_y[v] or mz < max_z[v] or 0 > min_z[v] ):
            print("v",vert_no[v])
            print("!!!!!!:",max(max_x[v] ,max_y[v] ,max_z[v]))
            print("!!!!!!:",min(min_x[v] ,min_y[v] ,min_z[v]))
            max_x[v] = 0
            ####今見つけた場所より頭の方の椎骨番号のmax_xはすべて０にする。
            for k in range(v+1,len(vert_no)):
                if (max_x[k] == 0 or max_y[k] == 0 or max_z[k] == 0):
                    continue
                print("k", vert_no[k])
                print("!!!!!!:", max(max_x[k], max_y[k], max_z[k]))
                print("!!!!!!:", min(min_x[k], min_y[k], min_z[k]))
                max_x[k] = 0
            break

    #################################################################
    #椎骨切り出し(切り出し範囲の計算で求めた値を使用)
    #################################################################
    vert_img = [0]*len(vert_no)
    ans_img = [0]*len(vert_no)
    for v in range(len(vert_no)):
        if(max_x[v]==0 or max_y[v]==0 or max_z[v]==0):
            continue
        vert_img[v] = inp[min_x[v]:max_x[v],min_y[v]:max_y[v],min_z[v]:max_z[v]]
        ans_img[v] = ans[min_x[v]:max_x[v],min_y[v]:max_y[v],min_z[v]:max_z[v]]

    #################################################################
    #サイズの正規化(vert_img→vert_img2, ans_img→ans_img2)
    #################################################################
    norm_start = time.time()
    norm_size = math.floor((MAX_SIZE/math.sqrt(2)) + SLB*2)###クロス面がMAX_SIZEになるように正規化する。
    vert_img2 = [0]*len(vert_no)
    ans_img2 = [0]*len(vert_no)
    for v in range(len(vert_no)):
        print("サイズの正規化: ",v)
        if (max_x[v] == 0 or max_y[v] == 0 or max_z[v] == 0):
            continue

        # --- GPU-based normalization for the CT volume ---
        # The old calculation of target_reso is no longer needed.
        vert_img2[v] = apply_normalization(vert_img[v], output_size=norm_size, interpolation_mode='trilinear', gpu_id=gpu_id)
        np.nan_to_num(vert_img2[v], copy=False)

        # --- GPU-based normalization for the answer mask ---
        # Each label is interpolated separately with 'nearest' to preserve mask integrity.
        lnum = np.unique(ans_img[v])[1:]
        lnum.sort()
        print("ans_img lnum", lnum)

        if len(lnum) > 0:
            # Create a stack of binary masks for each label
            resampled_masks = []
            for label in lnum:
                binary_mask = (ans_img[v] == label)
                resampled_mask = apply_normalization(binary_mask, output_size=norm_size, interpolation_mode='nearest', gpu_id=gpu_id)
                resampled_masks.append(resampled_mask)
            
            # Combine the resampled masks back into a single multi-label mask
            # Start with an empty volume
            combined_mask = np.zeros((norm_size, norm_size, norm_size), dtype=np.float32)
            for i, label in enumerate(lnum):
                # Use the resampled mask (it will be 0s and 1s) and multiply by the label value
                # Add it to the combined mask. Where masks overlap, the highest label value will win.
                combined_mask[resampled_masks[i] > 0.5] = label
            ans_img2[v] = combined_mask
        else:
            # If there are no labels, create an empty volume
            ans_img2[v] = np.zeros((norm_size, norm_size, norm_size), dtype=np.float32)

        print("norm_size",norm_size)
        print("vert_shape元", vert_img[v].shape)
        print("ans_shape元", ans_img[v].shape)
        print("vert_shape後",vert_img2[v].shape)
        print("ans_shape後", ans_img2[v].shape)
        print("x",len(vert_img2[v][:,0,0]))
        print("y",len(vert_img2[v][0,:,0]))
        print("z",len(vert_img2[v][0,0,:]))

    norm_end = time.time()
    print(f"[TIME] Normalization: {norm_end - norm_start:.2f}s")

    #################################################################
    # 不要なメモリ解放
    #################################################################
    del vert_img
    del ans_img
    gc.collect()

    #################################################################
    # NIIの保存（目的：答え合わせ）
    #################################################################
    for v in range(len(vert_no)):
        print("NIIの保存: ",v)
        if (max_x[v] == 0 or max_y[v] == 0 or max_z[v] == 0):
            continue
        sx = len(vert_img2[v][:,0,0])
        sy = len(vert_img2[v][0,:,0])
        sz = len(vert_img2[v][0,0,:])
        img_nii = nib.Nifti1Image(vert_img2[v][SLB:sx-SLB,SLB:sy-SLB,SLB:sz-SLB],affine=inp_struct.affine)
        text1 = cn_opath + str(vert_no[v]) + ".nii"
        nib.save(img_nii,text1)
        tmp = ans_img2[v][SLB:sx-SLB,SLB:sy-SLB,SLB:sz-SLB]
        lnum = np.unique(tmp)[1:]
        ans_nii = nib.Nifti1Image(tmp,affine=inp_struct.affine)
        text1 = an_opath + str(vert_no[v]) + "_" + str(lnum) + ".nii"
        nib.save(ans_nii,text1)

    #################################################################
    # CT回転＆スライス保存
    #################################################################
    rotate_start = time.time()
    original_size_cross = 0
    BONE_THR_MIN = 0  # minimum CT value of bone segmentation
    BONE_THR_MAX = 1900  # maximum CT value of bone segmentation
    for v in range(len(vert_no)):
        print("CT回転＆スライス保存: ",v)
        vert_img3 = [0 for j in range(3)]
        if (max_x[v] == 0 or max_y[v] == 0 or max_z[v] == 0):
            continue
        lnum = np.unique(ans_img2[v])[1:]
        print("ANS NII 回転前:", lnum)
        #### SgCr - Use GPU-accelerated rotation
        vert_img3[0] = gpu_rotate_3d(vert_img2[v], angle=45, axes=(0,1), reshape=True, gpu_id=gpu_id)
        print("SgCr", vert_img3[0].shape)
        ### CrAx - Use GPU-accelerated rotation
        vert_img3[1] = gpu_rotate_3d(vert_img2[v], angle=45, axes=(1,2), reshape=True, gpu_id=gpu_id)
        print("CrAx", vert_img3[1].shape)
        ### AxSg - Use GPU-accelerated rotation
        vert_img3[2] = gpu_rotate_3d(vert_img2[v], angle=45, axes=(0,2), reshape=True, gpu_id=gpu_id)
        print("AxSg", vert_img3[2].shape)

        #################################################################
        # CT スライス(スライス画像をvert_sliceに格納)
        #################################################################
        vert_slice = [[] for j in range(9)]
        print("CT NII :", vert_no[v])
        
        # Sagittal, Coronal, Axial slices from original volume
        tmp = vert_img2[v]
        print("CTスライス", tmp.shape)
        vert_slice[0] = [tmp[i, :, :] for i in range(tmp.shape[0])]
        vert_slice[1] = [tmp[:, i, :] for i in range(tmp.shape[1])]
        vert_slice[2] = [tmp[:, :, i] for i in range(tmp.shape[2])]

        # Slices from rotated volumes
        # axes=(0,1) -> SgCr
        tmp = vert_img3[0]
        print("CTスライスSgCr", tmp.shape)
        vert_slice[3] = [tmp[i, :, :] for i in range(tmp.shape[0])]
        vert_slice[4] = [tmp[:, i, :] for i in range(tmp.shape[1])]
        
        # axes=(1,2) -> CrAx
        tmp = vert_img3[1]
        print("CTスライスCrAx", tmp.shape)
        vert_slice[5] = [np.array(tmp[:, i, :]).T for i in range(tmp.shape[1])]
        vert_slice[6] = [np.array(tmp[:, :, i]).T for i in range(tmp.shape[2])]

        # axes=(0,2) -> AxSg
        tmp = vert_img3[2]
        print("CTスライスAxSg", tmp.shape)
        vert_slice[7] = [np.array(tmp[i, :, :]).T for i in range(tmp.shape[0])]
        vert_slice[8] = [tmp[:, :, i] for i in range(tmp.shape[2])]

        #################################################################
        # スライスのスラブ化＆周囲除去(vert_slice→vert_slice2) - Vectorized
        #################################################################
        vert_slice2 = [[] for j in range(9)]
        for j in range(9):
            if not vert_slice[j]:
                continue

            # リストを3D NumPy配列に変換
            slab_stack = np.array(vert_slice[j])

            print("スラブ化tmp[0]:", slab_stack[0].shape)
            stx, sty, stz = SLB, SLB, SLB
            edx, edy, edz = slab_stack.shape[2] - SLB, slab_stack.shape[1] - SLB, slab_stack.shape[0] - SLB

            if j >= 3:
                stx, stz = SLB2, SLB2
                edx, edz = slab_stack.shape[2] - SLB2, slab_stack.shape[0] - SLB2
                original_size_cross = slab_stack.shape[1]

            # Optimized vectorized slab computation
            num_slices = edz - stz
            cropped_stack = slab_stack[stz:edz, sty:edy, stx:edx]

            # Slab 1 (single slice) - already cropped
            tmp1 = [cropped_stack[i] for i in range(num_slices)]

            # Slab 15 (Average Intensity Projection) - vectorized
            tmp15 = []
            for i in range(num_slices):
                start, end = max(0, i - 7), min(num_slices, i + 8)
                slab_15 = np.mean(cropped_stack[start:end], axis=0)
                tmp15.append(slab_15)

            # Slab 31 (Average Intensity Projection) - vectorized
            tmp31 = []
            for i in range(num_slices):
                start, end = max(0, i - 15), min(num_slices, i + 16)
                slab_31 = np.mean(cropped_stack[start:end], axis=0)
                tmp31.append(slab_31)

            vert_slice2[j] = [tmp1, tmp15, tmp31]

        #################################################################
        # ピクセル値の正規化＆スライス保存(BONE_THR_MAX～BONE_THR_MINの値が２５５に入るように正規化)
        #################################################################
        for j in range(9):
            tmp = vert_slice2[j]
            print("ピクセル値の正規化＆スライス保存tmp:",[v,j,len(tmp)])
            sl=[0]*3
            for i in range(3):###スラブの数
                tmp2 = tmp[i]
                print("tmp2", len(tmp2))
                print("tmp2[0]", tmp2[0].shape)
                sl2 = []
                for k in range(len(tmp2)):###スライスの数
                    tmp3 = tmp2[k]
                    tmp3[tmp3 < BONE_THR_MIN] = BONE_THR_MIN
                    tmp3[tmp3 > BONE_THR_MAX] = BONE_THR_MAX
                    tmp3 = (((tmp3 - BONE_THR_MIN) / (BONE_THR_MAX - BONE_THR_MIN)) * 255)
                    tmp3 = np.asarray(tmp3, dtype='uint8')
                    sl2.append(tmp3)
                sl[i] = sl2

            print("sl[0]:", len(sl[0]))
            print("sl[1]:", len(sl[1]))
            print("sl[2]:", len(sl[2]))
            print("sl[0][0]:", sl[0][0].shape)
            print("sl[1][0]:", sl[1][0].shape)
            print("sl[2][0]:", sl[2][0].shape)
            text1 = si_opath + "AI" + f'{sbj_no:04}' + "_vert" + str(vert_no[v]) + PLANE_NAME[j] + "AVGProjectionIntensity0208\\"
            os.makedirs(text1, exist_ok=True)
            for k in range(len(sl[0])):
                tmp1 = sl[0][k]
                tmp15 = sl[1][k]
                tmp31 = sl[2][k]
                pil_img_gray = Image.fromarray(tmp1)
                img1 = pil_img_gray.convert('L')
                img1 = ImageOps.colorize(img1, black=(0, 0, 0), white=(255, 0, 0))
                pil_img_gray = Image.fromarray(tmp15)
                img2 = pil_img_gray.convert('L')
                img2 = ImageOps.colorize(img2, black=(0, 0, 0), white=(0, 255, 0))
                pil_img_gray = Image.fromarray(tmp31)
                img3 = pil_img_gray.convert('L')
                img3 = ImageOps.colorize(img3, black=(0, 0, 0), white=(0, 0, 255))
                imgc = ImageChops.add(img1, img2)
                imgc = ImageChops.add(imgc, img3)
                data = np.asarray(imgc)
                pil_imgc = Image.fromarray(data)
                text2 = text1 + "A" + f'{sbj_no:04}' + "_V" + str(vert_no[v]) + "_" +  f'{k:03}' + "_000.png"
                #print(pil_imgc.mode)
                pil_imgc.save(text2)

    rotate_end = time.time()
    print(f"[TIME] Rotation & Slicing: {rotate_end - rotate_start:.2f}s")
        # ss = int(norm_size*1.41/2)
        # fig, ax = plt.subplots()
        # ax.imshow(tmp[:, :, ss], cmap='gray', origin='lower')
        # ax.set_xlabel('y45')
        # ax.set_ylabel('x45')
        # plt.show()
        # fig, ax = plt.subplots()
        # ax.imshow(tmp[:, ss, :], cmap='gray', origin='lower')
        # ax.set_xlabel('z')
        # ax.set_ylabel('x45')
        # plt.show()
        # fig, ax = plt.subplots()
        # ax.imshow(tmp[ss, :, :], cmap='gray', origin='lower')
        # ax.set_xlabel('z')
        # ax.set_ylabel('y45')
        # plt.show()

    #################################################################
    # 椎骨切り出し位置の出力
    #################################################################
    text2 = ""
    for v in range(len(vert_no)):  ##椎骨番号
        print("椎骨切り出し位置の出力: ", v)
        if (max_x[v] == 0 or max_y[v] == 0 or max_z[v] == 0):
            continue
        ####スライス画像名の生成
        text2 = text2 + str(vert_no[v]) + "," + str(SLB) + "," + str(SLB2) + "," \
                + str(len(vert_img2[v][:, 0, 0])) + "," + str(original_size_cross) + "," + str(min_x[v]) + "," \
                + str(max_x[v]) + "," + str(min_y[v]) + "," + str(max_y[v]) + "," \
                + str(min_z[v]) + "," + str(max_z[v]) + ","
        # text2 = text2 + str(vert_no[v]) + "," + str(min_x[v]+SLB)+ "," + str(max_x[v]-SLB)+ "," \
        #        + str(min_y[v]+SLB) + "," + str(max_y[v]-SLB) + "," + str(min_z[v]+SLB) + "," + str(max_z[v]-SLB)
        text2 = text2 + "\n"
    text1 = inp_nii + "cut_li" + f'{sbj_no:04}' + ".txt"
    f = open(text1, 'w', encoding='UTF-8')
    f.write(text2)
    f.close()

    #################################################################
    # 答えの回転＆スライス化（答えの場合は、スライス切り出しも実施）
    #################################################################
    ans_slice2 = [[0 for i in range(len(vert_no))] for j in range(9)]
    for v in range(len(vert_no)):
        print("答えの回転＆スライス化: ",v)
        if (max_x[v] == 0 or max_y[v] == 0 or max_z[v] == 0):
            continue
        ans_slice2[0][v], ans_slice2[1][v], ans_slice2[2][v] = acs_cut(ans_img2, v, SLB)
        if(ans_slice2[0][v][0]):print("Sagit_ans",ans_slice2[0][v][0][0].shape)
        if(ans_slice2[1][v][0]):print("Coron_ans",ans_slice2[1][v][0][0].shape)
        if(ans_slice2[2][v][0]):print("Axial_ans",ans_slice2[2][v][0][0].shape)
        ans_slice2[3][v], ans_slice2[4][v] = sgcr_cut(ans_img2, v, SLB, SLB2)
        if(ans_slice2[3][v][0]):print("SgCr_ans",ans_slice2[3][v][0][0].shape)
        if(ans_slice2[4][v][0]):print("SgCr_ans",ans_slice2[4][v][0][0].shape)
        ans_slice2[5][v], ans_slice2[6][v] = crax_cut(ans_img2, v, SLB, SLB2)
        if(ans_slice2[5][v][0]):print("CrAx_ans",ans_slice2[5][v][0][0].shape)
        if(ans_slice2[6][v][0]):print("CrAx_ans",ans_slice2[6][v][0][0].shape)
        ans_slice2[7][v], ans_slice2[8][v] = axsg_cut(ans_img2, v, SLB, SLB2)
        if(ans_slice2[7][v][0]):print("AxSg_ans",ans_slice2[7][v][0][0].shape)
        if(ans_slice2[8][v][0]):print("AxSg_ans",ans_slice2[8][v][0][0].shape)
    # #################################################################
    # #
    # ##################################################################
    # rect_nii = [[0 for i in range(len(vert_no))] for j in range(9)]
    # for p in range(len(PLANE_NAME)):
    #     if p!=0:
    #         continue
    #     for v in range(len(vert_no)):
    #         if (max_x[v] == 0 or max_y[v] == 0 or max_z[v] == 0):
    #             continue
    #         tmp = ans_slice2[p][v][0]
    #         for s in range(len(tmp)):
    #             rect_nii[p][v][s, :, :] = tmp[s]
    #
    # for v in range(len(vert_no)):
    #     if (max_x[v] == 0 or max_y[v] == 0 or max_z[v] == 0):
    #         continue
    #     img_nii = nib.Nifti1Image(rect_nii[0][v][SLB:(SLB+181),SLB:(SLB+181),SLB:(SLB+181)],affine=inp_struct.affine)
    #     text1 = "C:\TotalSeg\\" + "out_"+str(v)+".nii"
    #     nib.save(img_nii,text1)

    #################################################################
    #　学習用矩形リストの生成
    #################################################################
    ans_slice3 = [[0 for i in range(len(vert_no))] for j in range(9)]
    for v in range(len(vert_no)):
        print("学習用矩形リストの生成: ",v)
        if ( max_x[v] == 0 or max_y[v] == 0 or max_z[v] == 0 ):
            continue
        for j in range(9):
            tmp = ans_slice2[j][v]
            vvv = []
            if len(tmp) == 0:##ラベルなし（骨折なし）
                continue
            print("ANS ラベルの数,スライスの数:",[v,j,len(tmp), len(tmp[0])])
            for k in range(len(tmp[0])):###スライスの数
                hhh = []
                for h in range(len(tmp)):  ###ラベルの数
                    tmp2 = tmp[h]
                    tmp3 = tmp2[k]
                    lnum = np.unique(tmp3)[1:]
                    #print("h:", h, "lnum:",lnum)
                    xxb = 0
                    xxs = 9999
                    yyb = 0
                    yys = 9999
                    if len(lnum) == 0:
                        continue
                    val = 0
                    val_cnt=0
                    for x in range(len(tmp3[0, :])):###yとxが直感の逆
                        for y in range(len(tmp3[:, 0])):
                            if(tmp3[y, x] > 0):
                                if(xxb < x):xxb = x
                                if(xxs > x):xxs = x
                                if(yyb < y):yyb = y
                                if(yys > y):yys = y
                                val = val + round(tmp3[y, x])#####スラブ作成時の重畳数
                                val_cnt = val_cnt+1
                    #print("椎骨:", v, "\t平面 No:", j, "\tスライス No:", k,"\tラベル:", h, "\t答え矩形:", [xxs,yys,xxb,yyb,h,val])
                    hhh.append([xxs,yys,xxb,yyb,h,int(val/val_cnt)])
                vvv.append(hhh)
            ans_slice3[j][v]=vvv

        ###答えスライスの出力
        for j in range(9):
            vvv = ans_slice3[j][v]##矩形座標
            tmp = copy.deepcopy(ans_slice2[j][v])  ####値を置き換えるのでDeepCopy
            text1 = si_opath_ans + "_AI" + f'{sbj_no:04}' + "_vert" + str(vert_no[v]) + PLANE_NAME[j] + "\\"
            os.makedirs(text1, exist_ok=True)
            for k in range(len(tmp[0])):  ###スライスの数
                vvv2 = vvv[k]##矩形座標
                tmp2 = tmp[0][k]
                tmp2[tmp2 > 0] = 255
                pil_img_gray = Image.fromarray(tmp2)
                img1 = pil_img_gray.convert('L')
                img2 = ImageOps.colorize(img1, black=(0, 0, 0), white=(255, 0, 0))
                img3 = ImageOps.colorize(img1, black=(0, 0, 0), white=(0, 0, 0))
                img4 = ImageOps.colorize(img1, black=(0, 0, 0), white=(0, 0, 0))
                for h in range(len(tmp) - 1):  ###ラベルの数
                    tmp2 = tmp[h + 1][k]
                    tmp2[tmp2 > 0] = 255
                    pil_img_gray = Image.fromarray(tmp2)
                    if (h % 2 == 0):
                        val = 255 - int(h / 2) * 50
                        img3 = pil_img_gray.convert('L')
                        img3 = ImageOps.colorize(img3, black=(0, 0, 0), white=(0, val, 0))
                    else:
                        val = 255 - int(h / 2) * 50
                        img4 = pil_img_gray.convert('L')
                        img4 = ImageOps.colorize(img4, black=(0, 0, 0), white=(0, 0, val))
                    if (h == 0):
                        imgc = ImageChops.add(img2, img3)
                    elif (h % 2 == 0):
                        imgc = ImageChops.add(imgc, img3)
                    else:
                        imgc = ImageChops.add(imgc, img4)
                if ((len(tmp) - 1) == 0):
                    imgc = ImageChops.add(img2, img3)
                    imgc = ImageChops.add(imgc, img4)

                #######
                data = np.asarray(imgc)
                pil_imgc = Image.fromarray(data)
                draw = ImageDraw.Draw(pil_imgc)
                for h in range(len(vvv2)):  ###ラベルの数##矩形座標
                    #draw.rectangle([(vvv2[h][0]-1, vvv2[h][1]-1), (vvv2[h][2]+1, vvv2[h][3]+1)], outline=(255, 201, 210), width=1, dash=(2, 3))  # 矩形の描画
                    draw.rectangle([(vvv2[h][0], vvv2[h][1]), (vvv2[h][2], vvv2[h][3])], outline=(255, 201, 210), width=1)  # 矩形の描画
                text2 = text1 + "ans" + f'{sbj_no:04}' + "_V" + str(vert_no[v]) + "_" + f'{k:03}' + "_000.png"
                pil_imgc.save(text2)

        ###矩形付きスライスの出力
        for j in range(9):
            vvv = ans_slice3[j][v]  ##矩形座標
            text1 = si_opath_rect + "rAI" + f'{sbj_no:04}' + "_vert" + str(vert_no[v]) + PLANE_NAME[j] + "\\"
            os.makedirs(text1, exist_ok=True)
            for k in range(len(vvv)):  ###スライスの数
                vvv2 = vvv[k]  ##矩形座標
                rgb_slice = si_opath + "AI" + f'{sbj_no:04}' + "_vert" + str(vert_no[v]) + PLANE_NAME[j] + "AVGProjectionIntensity0208\\"
                rgb_slice2 = rgb_slice + "A" + f'{sbj_no:04}' + "_V" + str(vert_no[v]) + "_" + f'{k:03}' + "_000.png"
                img = Image.open(rgb_slice2)  #
                draw = ImageDraw.Draw(img)  # 矩形の描画の準備
                for h in range(len(vvv2)):  ###ラベルの数##矩形座標
                    #draw.rectangle([(vvv2[h][0]-1, vvv2[h][1]-1), (vvv2[h][2]+1, vvv2[h][3]+1)], outline=(255, 201, 210), width=1, dash=(2, 3))  # 矩形の描画
                    draw.rectangle([(vvv2[h][0], vvv2[h][1]), (vvv2[h][2], vvv2[h][3])], outline=(255, 201, 210), width=1)  # 矩形の描画
                text2 = text1 + "rect" + f'{sbj_no:04}' + "_V" + str(vert_no[v]) + "_" + f'{k:03}' + "_000.png"
                img.save(text2)

    #################################################################
    #　学習用矩形リストの出力
    #################################################################
    for v in range(len(vert_no)):##椎骨番号
        print("学習用矩形リストの出力: ",v)
        if ( max_x[v] == 0 or max_y[v] == 0 or max_z[v] == 0 ):
            continue
        for j in range(9):##断面方向
            tmp = ans_slice3[j][v]
            text2 = ""
            for k in range(len(tmp)):  ###スライスの数
                ####スライス画像名の生成
                slice_name1 = si_opath + "AI" + f'{sbj_no:04}' + "_vert" + str(vert_no[v]) + PLANE_NAME[j] + "AVGProjectionIntensity0208\\"
                slice_name2 = slice_name1 + "A" + f'{sbj_no:04}' + "_V" + str(vert_no[v]) + "_" + f'{k:03}' + "_000.png"
                if len(tmp[k]) <= 0:#矩形がない場合
                    text2 = text2 + slice_name2
                else:
                    text2 = text2 + slice_name2 + " "
                #####
                tmp2 = tmp[k]
                for h in range(len(tmp2)):  ###ラベルの数
                    #print("椎骨:",v,"\t平面 No:",j,"\tスライス No:",k,"\t答え矩形:",tmp2[h])
                    for i in range(len(tmp2[h])):
                        text2 = text2 + str(tmp2[h][i])
                        if( i != len(tmp2[h])-1):
                            text2 = text2 + ","
                    if (h != len(tmp2)-1):
                        text2 = text2 + " "
                text2 = text2 + "\n"
            text1 = al_opath + str(vert_no[v]) + "_" + str(j) + ".txt"
            f = open(text1, 'w', encoding='UTF-8')
            f.write(text2)
            f.close()
            text1 = al_opath2 + "_" + str(j) + ".txt"
            f = open(text1, 'a', encoding='UTF-8')
            f.write(text2)
            f.close()

            print("学習用矩形リストの出力: ",text1)

    #################################################################
    # 不要なメモリ解放
    #################################################################
    del vert_img2
    del ans_img2
    del ans_slice2
    del ans_slice3
    gc.collect()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[INFO] Processing subject {sbj_no} - Complete")
    print(f"[TIME] Total processing time: {total_time:.2f}s ({total_time/60:.2f}min)")
