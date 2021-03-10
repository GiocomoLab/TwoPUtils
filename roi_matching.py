import os
import math

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import suite2p as s2p
from suite2p.registration import bidiphase, utils, rigid, nonrigid


def align_roistack(roistack, rigid_offsets, nonrigid_offsets, ops):
    """

    :param roistack:
    :param rigidOffsets:
    :param nonrigidOffsets:
    :param ops:
    :return:
    """

    ymax, xmax = rigid_offsets
    ymax1, xmax1 = nonrigid_offsets
    for roi in range(roistack.shape[0]):
        roistack[roi, :, :] = rigid.shift_frame(frame=roistack[roi, :, :], dy=ymax, dx=xmax)

        roistack[roi, :, :] = nonrigid.transform_data(
            data=roistack[roi:roi + 1, :, :, ],
            nblocks=ops['nblocks'],
            xblock=ops['xblock'],
            yblock=ops['yblock'],
            ymax1=ymax1,
            xmax1=xmax1,
        )


def align_stack(ref_img, frames, ops):
    """

    :param refImg:
    :param frames:
    :param ops:
    :return:
    """

    rmin, rmax = np.percentile(ref_img, 1), np.percentile(ref_img, 99)
    ref_img = np.clip(ref_img, rmin, rmax).astype(np.float32)

    mask_mul, mask_offset = rigid.compute_masks(
        refImg=ref_img,
        maskSlope=ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma'],
    )

    cfRefImg = rigid.phasecorr_reference(
        refImg=ref_img,
        smooth_sigma=ops['smooth_sigma'],
        pad_fft=ops['pad_fft'],
    )

    if ops.get('nonrigid'):
        if 'yblock' not in ops:
            ops['yblock'], ops['xblock'], ops['nblocks'], ops['block_size'], ops[
                'NRsm'] = nonrigid.make_blocks(Ly=ops['Ly'], Lx=ops['Lx'], block_size=ops['block_size'])

        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(
            refImg0=ref_img,
            maskSlope=ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma'],
            # slope of taper mask at the edges
            smooth_sigma=ops['smooth_sigma'],
            yblock=ops['yblock'],
            xblock=ops['xblock'],
            pad_fft=ops['pad_fft'],
        )

    # ### ------------- register binary to reference image ------------ ##

    fsmooth = frames.copy().astype(np.float32)
    if ops['smooth_sigma_time'] > 0:
        fsmooth = utils.temporal_smooth(data=fsmooth, sigma=ops['smooth_sigma_time'])

    # rigid registration
    if ops.get('norm_frames', False):
        fsmooth = np.clip(fsmooth, rmin, rmax)

    ymax, xmax, cmax = rigid.phasecorr(
        data=rigid.apply_masks(data=fsmooth, maskMul=mask_mul, maskOffset=mask_offset),
        cfRefImg=cfRefImg,
        maxregshift=ops['maxregshift'],
        smooth_sigma_time=ops['smooth_sigma_time'],
    )
    rigidOffsets = [ymax, xmax]

    # for frame, frame_chan2, dy, dx in zip(frames, frames_chan2, ymax, xmax):
    for frame, dy, dx in zip(frames, ymax, xmax):
        frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)

    # non-rigid registration
    if ops['nonrigid']:
        # need to also shift smoothed data (if smoothing used)
        if ops['smooth_sigma_time'] or ops['1Preg']:
            for fsm, dy, dx in zip(fsmooth, ymax, xmax):
                fsm[:] = rigid.shift_frame(frame=fsm, dy=dy, dx=dx)
        else:
            fsmooth = frames.copy()

        if ops.get('norm_frames', False):
            fsmooth = np.clip(fsmooth, rmin, rmax)

        ymax1, xmax1, cmax1 = nonrigid.phasecorr(
            data=fsmooth,
            maskMul=maskMulNR.squeeze(),
            maskOffset=maskOffsetNR.squeeze(),
            cfRefImg=cfRefImgNR.squeeze(),
            snr_thresh=ops['snr_thresh'],
            NRsm=ops['NRsm'],
            xblock=ops['xblock'],
            yblock=ops['yblock'],
            maxregshiftNR=10,
        )

        frames = nonrigid.transform_data(
            data=frames,
            nblocks=ops['nblocks'],
            xblock=ops['xblock'],
            yblock=ops['yblock'],
            ymax1=ymax1,
            xmax1=xmax1,
        )

    nonrigidOffsets = [ymax1, xmax1]

    return frames, rigidOffsets, nonrigidOffsets

def make_roistack(stats, ops):
    """

    :param stats:
    :param ops:
    :return:
    """
    roistack = np.zeros([stats.shape[0], ops['Ly'], ops['Lx']]).astype(np.float32)
    for i, roi in enumerate(stats):
        roistack[i, roi['ypix'], roi['xpix']] = 1
    return roistack

def get_com_from_roistack(roistack):
    """


    :param roistack:
    :return:
    """
    com = np.zeros([roistack.shape[0], 2])
    for ind in range(roistack.shape[0]):
        com[ind, :] = np.argwhere(roistack[ind, :, :]).mean(axis=0)
    return com

