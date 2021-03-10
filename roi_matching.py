import os
import math
from functools import reduce

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from suite2p.registration import bidiphase, rigid, nonrigid
from suite2p.registration import utils as reg_utils


class ROIAligner:

    def __init__(self, ref_sess, targ_sess):
        """

        :param ref_sess:
        :param targ_sess:
        """

        self.ref_sess = ref_sess
        if isinstance(targ_sess, list) or isinstance(targ_sess, tuple):
            self.targ_sess = targ_sess
        else:
            self.targ_sess = (targ_sess)

        self.ref_roistack = TwoPUtils.roi_matching.make_roistack(self.ref_sess.s2p_stats, self.ref_sess.s2p_ops)

        self.ref_match_inds = [[]] * len(self.targ_sess)
        self.targ_match_inds = [[]] * len(self.targ_sess)

        self.align_mean_images()

    def run_pairwise_matches(self):
        """

        :return:
        """

        for i in range(len(self.targ_sess)):
            self.match_session_pair(i)

    def align_mean_images(self):
        """

        :return:
        """

        # get mean images
        self.frames = np.array([s.s2p_ops['meanImg'] for s in self.targ_sess]).astype(np.float32)
        # align them
        self.frames, self.rigidOffsets, self.nonrigidOffsets = TwoPUtils.roi_matching.align_stack(
            self.ref_sess.s2p_ops['meanImg'], self.frames, self.ref_sess.s2p_ops)

    def match_session_pair(self, index, plot_iou_hist=True):
        """

        :param index:
        :param plot_iou_hist:
        :return:
        """

        targ_roistack = make_roistack(self.targ_sess[index].s2p_stats,
                                                             self.targ_sess[index].s2p_ops)

        # apply transform to roi masks
        self.align_roistack(targ_roistack, [self.rigidOffsets[0][index], self.rigidOffsets[1][index]],
                                              [self.nonrigidOffsets[0][index:index + 1, :],
                                               self.nonrigidOffsets[1][index:index + 1, :]],
                                              self.ref_sess.s2p_ops)

        # calculate com from roistack
        com_ref = get_com_from_roistack(self.ref_roistack)
        com_targ = get_com_from_roistack(targ_roistack)

        # reduce candidate matches for speed
        dist = np.linalg.norm(com_ref[:, np.newaxis, :] - com_targ[np.newaxis, :, :], ord=2, axis=-1)
        candidates = dist < 10

        # calculate intersection over union for candidate matches
        iou = self.iou(self.ref_roistack, targ_roistack, candidates)

        # calculate iou threshold
        thresh = self.set_iou_threshold(iou)

        # get matched rois
        ref_match_inds, targ_match_inds = self.get_matches(iou, thresh)
        self.ref_match_inds[index], self.targ_match_inds[index] = ref_match_inds, targ_match_inds

    @property
    def common_rois_all_sessions(self):
        """

        :return:
        """

        # find cells that are in match list each time
        ref_common_rois = reduce((lambda x, y: set(x) & set(y)), self.ref_match_inds)

        common_roi_mapping = {}
        for i, roi in enumerate(ref_common_rois):
            print(i, roi)
            self.common_roi_mapping[roi] = []
            for j, (ref_list, targ_list) in enumerate(zip(sa.ref_match_inds, sa.targ_match_inds)):
                print(j)
                ind = np.argwhere(ref_list == roi)[0]
                assert ind.shape[0] == 1, "duplicate matched rois somewhere"

                common_roi_mapping[roi].append(targ_list[ind[0]])

        return common_roi_mapping

    @staticmethod
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

    @staticmethod
    def iou(ref_roistack, targ_roistack, candidates):
        """

        :param ref_roistack:
        :param targ_roistack:
        :param candidates:
        :return:
        """

        ref_roistack_f = np.reshape(ref_roistack, [ref_roistack.shape[0], -1])
        targ_roistack_f = np.reshape(targ_roistack, [targ_roistack.shape[0], -1])
        iou = np.zeros([ref_roistack.shape[0], targ_roistack.shape[0]])
        for i in range(ref_roistack.shape[0]):
            intxn = (ref_roistack_f[i:i + 1, :] * targ_roistack_f[candidates[i, :], :]).sum(axis=-1)
            union = (ref_roistack_f[i:i + 1, :] + targ_roistack_f[candidates[i, :], :]).sum(axis=-1)
            iou[i, candidates[i, :]] = intxn / union
        return iou

    @staticmethod
    def set_iou_threshold(iou):
        """

        :param iou:
        :return:
        """

        _iou = np.copy(iou)
        # find max across columns for each row and set it to 0
        _iou[:, np.argmax(_iou, axis=1)] = 0
        # find max again (2nd largest value per row)
        # set thresh to this number
        thresh = np.amax(_iou) + 1E-3
        return thresh

    @staticmethod
    def get_matches(iou, thresh):
        """

        :param iou:
        :param thresh:
        :return:
        """

        row_sort, col_sort = np.unravel_index(np.argsort(iou, axis=None), iou.shape)
        matched_ref, matched_targ = [], []

        for i, j in zip(row_sort[::-1], col_sort[::-1]):
            if (i not in matched_ref) and (j not in matched_targ) and (iou[i, j] > thresh):
                matched_ref.append(i)
                matched_targ.append(j)

        return matched_ref, matched_targ






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
        fsmooth = reg_utils.temporal_smooth(data=fsmooth, sigma=ops['smooth_sigma_time'])

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

