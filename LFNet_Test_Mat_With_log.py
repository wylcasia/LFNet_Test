"""
LFNet_Test
Author: Yunlong Wang
Date: 2018.01
"""
from __future__ import print_function, absolute_import
import os
import time
import datetime
import numpy as np
import scipy.io as sio
import gc

import skimage.io as io
import skimage.color as color
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

from src.utils.ioutils import opts_parser, del_files, mkdir_p, isfile, isdir, join
from src.utils.lfutils import FolderTo4DLF, ImgTo4DLF
from src.models import LFNet, load_model, config

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()


def test_LFNet(
        path = None,
        model_path = None,
        save_path = None,
        scene_names = None,
        train_length = 7,
        crop_length = 7,
        factor = 3,
        weight_row = 0.5,
        save_results = False
):

    options = locals().copy()

    if path is not None:
        log.info('='*40)
        if not isdir(path):
            raise IOError('No such folder: {}'.format(path))
        if save_path is None:
            save_path = path+'_eval_l%d_f%d/'%(crop_length,factor)
        if not isdir(save_path):
            log.warning('No such path for saving Our results, creating dir {}'
                        .format(save_path))
            mkdir_p(save_path)

        sceneNameTuple = tuple(scene_names)
        sceneNum = len(sceneNameTuple)

        if sceneNum == 0:
            raise IOError('No %s scene name in path %s' %(ext,eval_path))

    else:
        raise NameError('No folder given.')



    log_file = join(save_path,'LFNet_Test.log')
    if isfile(log_file):
        print('%s exists, delete it and rewrite...' % log_file)
        os.remove(log_file)
    fh = logging.FileHandler(log_file)
    log.addHandler(fh)

    log.info('='*40)
    log.info('Time Stamp: %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    total_PSNR = []
    total_SSIM = []
    total_Elapsedtime = []

    performacne_index_file = join(save_path,'performance_stat.mat')

    options['path'] = path
    options['Scenes'] = sceneNameTuple
    options['model_path'] = model_path
    options['save_path'] = save_path
    options['factor'] = factor
    options['train_length'] = train_length
    options['crop_length'] = crop_length
    options['save_results'] = save_results

    rn_model_file = 'LFNet_RN_with_IMsF_f%d_l%d.npy' %(factor,train_length)
    if not isfile(join(model_path,rn_model_file)):
        raise IOError('No Such Model File %s', join(model_path,rn_model_file))
    else:
        log.info('Loading pre-trained Row Network model from %s' % join(model_path,rn_model_file))
        rn_model = load_model(join(model_path,rn_model_file))

    cn_model_file = 'LFNet_CN_with_IMsF_f%d_l%d.npy' %(factor,train_length)
    if not isfile(join(model_path,cn_model_file)):
        raise IOError('No Such Model File %s', join(model_path,cn_model_file))
    else:
        log.info('Loading pre-trained Column Network model from %s' % join(model_path,cn_model_file))
        cn_model = load_model(join(model_path,cn_model_file))

    c_imsf = rn_model['f_IMsF_0_w'].shape[0]
    c_in_imsf = rn_model['f_IMsF_0_w'].shape[1]
    k0_imsf = rn_model['f_IMsF_0_w'].shape[-1]
    k1_imsf = rn_model['f_IMsF_1_w'].shape[-1]
    k2_imsf = rn_model['f_IMsF_2_w'].shape[-1]
    k3_imsf = rn_model['f_IMsF_3_w'].shape[-1]

    c1 = rn_model['f_conv_0_v_w'].shape[0]
    c1_in = rn_model['f_conv_0_v_w'].shape[1]
    k1 = rn_model['f_conv_0_v_w'].shape[-1]
    c2 = rn_model['f_conv_1_v_w'].shape[0]
    k2 = rn_model['f_conv_1_v_w'].shape[-1]
    k3 = rn_model['f_conv_2_v_w'].shape[-1]

    c0_r = rn_model['f_conv_0_r_w'].shape[0]
    k0_r = rn_model['f_conv_0_r_w'].shape[-1]
    c1_r = rn_model['f_conv_1_r_w'].shape[0]
    k1_r = rn_model['f_conv_1_r_w'].shape[-1]

    options['IMsF_shape'] = [
        [c_imsf, c_in_imsf, k0_imsf, k0_imsf],
        [c_imsf, c_imsf, k1_imsf, k1_imsf],
        [c_imsf, c_imsf, k2_imsf, k2_imsf],
        [c_imsf, c_imsf, k3_imsf, k3_imsf]
    ]

    options['filter_shape'] = [
        [c1, c1_in, k1, k1],
        [c2, c1, k2, k2],
        [c_in_imsf, c2, k3, k3]
    ]
    options['rec_filter_size'] = [
        [c0_r, c0_r, k0_r, k0_r],
        [c1_r, c1_r, k1_r, k1_r]
    ]

    # options['IMsF_shape'] = [
    #     [64, 1, 3, 3],
    #     [64, 64, 3, 3],
    #     [64, 64, 3, 3],
    #     [64, 64, 3, 3]
    # ]
    #
    # options['filter_shape'] = [
    #     [64, 64, 5, 5],
    #     [32, 64, 1, 1],
    #     [1, 32, 9, 9]
    # ]
    # options['rec_filter_size'] = [
    #     [64, 64, 1, 1],
    #     [32, 32, 1, 1]
    # ]

    # options['padding'] = np.sum([(i[-1] - 1) / 2 for i in options['filter_shape']])
    # diff = options['padding']

    log.info('='*40)
    log.info("model options\n"+str(options))

    log.info('='*40)
    tic = time.time()
    log.info('... Building pre-trained model' )
    net = LFNet(options)
    rn_x = net.build_net(rn_model)
    cn_x = net.build_net(cn_model)
    log.info("Elapsed time: %.2f sec" % (time.time() - tic))

    for scene in sceneNameTuple:
        log.info('='*15+scene+'='*15)
        if save_results:
            our_save_path = join(save_path,scene + '_OURS')
            GT_save_path = join(save_path,scene + '_GT')
            if isdir(our_save_path):
                log.info('='*40)
                del_files(our_save_path,'png')
                log.warning('Ours Save Path %s exists, delete all .png files' % our_save_path)
            else:
                mkdir_p(our_save_path)

            if isdir(GT_save_path):
                log.info('='*40)
                del_files(GT_save_path,'png')
                log.warning('GT path %s exists, delete all .png files' % GT_save_path)
            else:
                mkdir_p(GT_save_path)

        if isfile(os.path.join(path,scene+'.mat')):
            log.info('='*40)
            log.info('Loading GT and LR data from %s' % os.path.join(path,scene+'.mat'))
            dump = sio.loadmat(os.path.join(path,scene+'.mat'))
        else:
            raise IOError('No such .mat file: %s' % os.path.join(path,scene+'.mat'))

        lf = dump['gt_data'].astype(config.floatX)
        lr_lf = dump['lr_data'].astype(config.floatX)



        input_lf = lf[:,:,0,:,:]
        x_res = input_lf.shape[0]
        y_res = input_lf.shape[1]
        s_res = input_lf.shape[2]
        t_res = input_lf.shape[3]

        output_lf = np.zeros((x_res,y_res,s_res,t_res)).astype(config.floatX)

        log.info('='*40)
        s_time = time.time()
        log.info('LFNet SR running.....')
        log.info('>>>> Row Network')
        for s_n in range(s_res):
            row_seq = np.transpose(lr_lf[:,:,s_n,:],(2,0,1))
            up_row_seq = rn_x(row_seq[:,np.newaxis,np.newaxis,:,:])
            output_lf[:,:,s_n,:] += np.transpose(up_row_seq[:,0,0,:,:],(1,2,0)) * weight_row
        log.info('>>>> Column Network')
        for t_n in range(t_res):
            col_seq = np.transpose(lr_lf[:,:,:,t_n],(2,0,1))
            up_col_seq = cn_x(col_seq[:,np.newaxis,np.newaxis,:,:])
            output_lf[:,:,:,t_n] += np.transpose(up_col_seq[:,0,0,:,:],(1,2,0)) * (1 - weight_row)
        process_time = time.time() - s_time
        log.info('Elapsed Time: %.2f sec per view'
                 % (process_time/(s_res*t_res)))

        PSNR = []
        SSIM = []

        log.info('='*40)
        log.info('Evaluation......')
        log.info('Predicted LF shape: %s' % str(output_lf.shape))
        log.info('GT LF shape: %s' % str(lf.shape))
        log.info('='*40)

        for s_n in xrange(s_res):
            for t_n in xrange(t_res):

                gt_img = np.clip(lf[:,:,0,s_n,t_n],16.0/255.0,235.0/255.0)
                view_img = np.clip(output_lf[:,:,s_n,t_n],16.0/255.0,235.0/255.0)

                this_PSNR = psnr(view_img,gt_img)
                this_SSIM = ssim(view_img,gt_img)

                # this_PSNR = psnr(np.uint8(view_img*255.0),np.uint8(gt_img*255.0))
                # this_SSIM = ssim(np.uint8(view_img*255.0),np.uint8(gt_img*255.0))

                log.info('View %.2d_%.2d: PSNR: %.2fdB SSIM: %.4f' %(s_n+1, t_n+1, this_PSNR, this_SSIM))

                PSNR.append(this_PSNR)
                SSIM.append(this_SSIM)

                if save_results:
                    filename = join(our_save_path,'View_'+str(s_n+1)+'_'+str(t_n+1)+'.png')
                    GTname = join(GT_save_path,'View_'+str(s_n+1)+'_'+str(t_n+1)+'.png')
                    out_img = np.zeros((x_res,y_res,3))
                    gt_out_img = np.zeros((x_res,y_res,3))

                    out_img[:,:,0] = np.clip(view_img*255.0,16.0,235.0)
                    gt_out_img[:,:,0] = np.clip(gt_img*255.0,16.0,235.0)
                    # print('Max: %.2f Min: %.2f' %(out_img[:,:,0].max(),out_img[:,:,0].min()))
                    out_img[:,:,1:3] = lf[:,:,1:3,s_n,t_n]*255.0
                    gt_out_img[:,:,1:3] = lf[:,:,1:3,s_n,t_n]*255.0
                    # print('Max: %.2f Min: %.2f' %(out_img[:,:,1].max(),out_img[:,:,1].min()))

                    out_img = color.ycbcr2rgb(out_img)
                    out_img = np.clip(out_img,0.0,1.0)
                    out_img = np.uint8(out_img*255.0)

                    gt_out_img = color.ycbcr2rgb(gt_out_img)
                    gt_out_img = np.clip(gt_out_img,0.0,1.0)
                    gt_out_img = np.uint8(gt_out_img*255.0)

                    io.imsave(filename,out_img)
                    io.imsave(GTname,gt_out_img)


        log.info('='*40)
        total_PSNR.append(np.mean(np.array(PSNR)))
        total_SSIM.append(np.mean(np.array(SSIM)))
        total_Elapsedtime.append((process_time/(s_res*t_res)))
        log.info('[PSNR] Min: %.2f Avg: %.2f Max: %.2f dB' %(np.min(np.array(PSNR)),
                                                         np.mean(np.array(PSNR)),
                                                         np.max(np.array(PSNR))))
        log.info('[SSIM] Min: %.4f Avg: %.4f Max: %.4f' %(np.min(np.array(SSIM)),
                                                         np.mean(np.array(SSIM)),
                                                         np.max(np.array(SSIM))))
        log.info("[Elapsed time] %.2f sec per view." % (process_time/(s_res*t_res)))
        gc.collect()
        log.info('='*40)


    log.info('='*3+'Average Performance on %d scenes' % len(sceneNameTuple)+'='*6)
    log.info('PSNR: %.2f dB' % np.mean(np.array(total_PSNR)))
    log.info('SSIM: %.4f' % np.mean(np.array(total_SSIM)))
    log.info('Elapsed Time: %.2f sec per view' % np.mean(np.array(total_Elapsedtime)))
    log.info('='*40)

    embeded = dict(NAME=sceneNameTuple,PSNR=np.array(total_PSNR),SSIM=np.array(total_SSIM),TIME=np.array(total_Elapsedtime))
    sio.savemat(performacne_index_file,embeded)

if __name__ == '__main__':

    parser = opts_parser()
    args = parser.parse_args()

    test_LFNet(path=args.path,model_path=args.model_path,factor=args.factor, train_length=args.train_length,
               crop_length=args.crop_length, scene_names=args.scenes, save_results=args.save_results,
               weight_row=args.weight_row)