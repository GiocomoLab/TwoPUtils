import os
import math
from functools import reduce

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from suite2p.registration import bidiphase, rigid, nonrigid
from suite2p.registration import utils as reg_utils


class ROIAligner:
    """
    Class for aligning ROIs across imaging sessions. This code assumes that the ROIs have been curated for
    each session already.
    """

    def __init__(self, ref_sess, targ_sess):
        """


        :param ref_sess: TwoPUtils.sess.Session instance for the session that all other sessions will be aligned to
        :param targ_sess: (list or tuple of) TwoPUtils.sess.Session instance(s) for the sessions that will be aligned
        """

        self.ref_sess = ref_sess
        # make targ_sess iterable if not
        if isinstance(targ_sess, list) or isinstance(targ_sess, tuple):
            self.targ_sess = targ_sess
        else:
            self.targ_sess = tuple(targ_sess)

        # roi x Ly x Lx matrix of ROI masks
        self.ref_roistack = make_roistack(self.ref_sess.s2p_stats, self.ref_sess.s2p_ops)

        # allocate
        self.ref_match_inds = [[]] * len(self.targ_sess)
        self.targ_match_inds = [[]] * len(self.targ_sess)
        self.frames = None
        self.rigid_offsets = None
        self.nonrigid_offsets = None

        # align targ_sess mean images to ref_sess mean image
        self.align_mean_images()

    def run_pairwise_matches(self):
        """
        calculate matched ROIs for ref_sess vs each targ_sess
        :return:
        """

        for i in range(len(self.targ_sess)):
            self.match_session_pair(i)

    def align_mean_images(self, **ops_kwargs):
        """
        Align targ_sess mean images to ref_sess mean image
        :param ops_kwargs: optional registration arguments, default is to use ref_sess.s2p_ops alignment parameters
        :return:
        """

        # update ops
        reg_ops = self.ref_sess.s2p_ops
        reg_ops.update(ops_kwargs)

        # get mean images
        self.frames = np.array([s.s2p_ops['meanImg'] for s in self.targ_sess]).astype(np.float32)
        # align them
        self.frames, self.rigid_offsets, self.nonrigid_offsets = align_stack(
            self.ref_sess.s2p_ops['meanImg'], self.frames, reg_ops)

    def match_session_pair(self, index, thresh = None):
        """
        Find matching ROI pairs for ref_sess and targ_sess[index]


        :param index: int, which target session to align
        :param thresh: float, iou threshold to run matching
        :return:
        """

        # make roi x Ly x Lx array of masks
        targ_roistack = make_roistack(self.targ_sess[index].s2p_stats,
                                      self.targ_sess[index].s2p_ops)

        # apply alignment transform to roi masks
        self.align_roistack(targ_roistack, [self.rigid_offsets[0][index], self.rigid_offsets[1][index]],
                            [self.nonrigid_offsets[0][index:index + 1, :],
                             self.nonrigid_offsets[1][index:index + 1, :]],
                            self.ref_sess.s2p_ops)

        # calculate center of mass of each roi from roistack
        com_ref = get_com_from_roistack(self.ref_roistack)
        com_targ = get_com_from_roistack(targ_roistack)

        # reduce candidate matches for speed
        dist = np.linalg.norm(com_ref[:, np.newaxis, :] - com_targ[np.newaxis, :, :], ord=2, axis=-1)
        candidates = dist < 10

        # calculate intersection over union (iou) for candidate matches
        iou = self.iou(self.ref_roistack, targ_roistack, candidates)

        # calculate iou threshold
        if thresh is None:
            thresh = self.set_iou_threshold(iou)

        # get matched rois
        ref_match_inds, targ_match_inds = self.get_matches(iou, thresh)
        self.ref_match_inds[index], self.targ_match_inds[index] = ref_match_inds, targ_match_inds

    @property
    def common_rois_all_sessions(self):
        """
        find ROIs common to all sessions
        :return common_roi_mapping: dict, keys are roi index from ref_sess, values are a list of matching roi indices
                                          in each session of targ_sess
        """

        # find cells that are in reference match list each time
        ref_common_rois = reduce((lambda x, y: list(set(x) & set(y))), self.ref_match_inds)

        # find matching indices
        common_roi_mapping = {}
        for i, roi in enumerate(ref_common_rois):
            common_roi_mapping[roi] = []
            for j, (ref_list, targ_list) in enumerate(zip(self.ref_match_inds, self.targ_match_inds)):
                print(j)
                ind = np.argwhere(ref_list == roi)[0]
                assert ind.shape[0] == 1, "duplicate matched rois somewhere"

                common_roi_mapping[roi].append(targ_list[ind[0]])

        return common_roi_mapping

    @staticmethod
    def align_roistack(roistack, rigid_offsets, nonrigid_offsets, ops):
        """
        Align each roi using rigid followed by nonrigid transforms
        :param roistack: np.array, float32, [rois, Ly, Lx], of binary roi masks
        :param rigid_offsets: list, [y_shift, x_shift] from self.align_mean_images
        :param nonrigid_offsets: list, [y_shift, x_shift] from self.align_mean_images
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
        intersection over union

        :param ref_roistack: np.array, [rois, Ly, Lx], reference session roi masks
        :param targ_roistack: np.array, [rois, Ly, Lx], target session roi masks
        :param candidates: np.array, [ref rois, target rois], boolean mask of candidate matches to reduce computation
        :return: iou, np.array, [ref rois, target rois], interscection over overlap of each roi pair
        """

        # flatten arrays
        ref_roistack_f = np.reshape(ref_roistack, [ref_roistack.shape[0], -1])
        targ_roistack_f = np.reshape(targ_roistack, [targ_roistack.shape[0], -1])

        iou = np.zeros([ref_roistack.shape[0], targ_roistack.shape[0]])
        for i in range(ref_roistack.shape[0]):
            # intersection
            intxn = (ref_roistack_f[i:i + 1, :] * targ_roistack_f[candidates[i, :], :]).sum(axis=-1)
            # union
            union = (ref_roistack_f[i:i + 1, :] + targ_roistack_f[candidates[i, :], :]).sum(axis=-1)
            iou[i, candidates[i, :]] = intxn / union
        return iou

    @staticmethod
    def set_iou_threshold(iou):
        """
        set iou threshold to guarantee unique matches

        :param iou: np.array, [ref rois, target rois], intersection over union values
        :return: thresh, np.float
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
        find matched rois

        :param iou: [ref rois, target rois], intersection over union values
        :param thresh: np.float, threshold to consider matches
        :return: matched_ref: list, roi indices in ref_sess of matches
                matched_targ: list, roi indices in targ_sess of matches
        """

        # sort iou's
        row_sort, col_sort = np.unravel_index(np.argsort(iou, axis=None), iou.shape)
        matched_ref, matched_targ = [], []

        # starting with highest overlap
        for i, j in zip(row_sort[::-1], col_sort[::-1]):
            # if not previously matched and iou>thresh
            if (i not in matched_ref) and (j not in matched_targ) and (iou[i, j] > thresh):
                matched_ref.append(i)
                matched_targ.append(j)

        return matched_ref, matched_targ


def align_stack(ref_img, frames, ops):
    """
    code stolen from suite2p to apply rigid followed by nonrigid alignment to align frames to ref_img using ops params
    :param ref_img: np.array, [Ly, Lx] reference image
    :param frames: np.array. [nframes, Ly, Lx] stack of target images to be aligned
    :param ops: dict, alignment parameters

    :return: frames: aligned input
            rigidOffsets: [y_shifts, x_shifts] from rigid motion correction
            nonrigidOffsets: [y_shifts, x_shifts] from nonrigid motion correction
    """

    # clip reference
    rmin, rmax = np.percentile(ref_img, 1), np.percentile(ref_img, 99)
    ref_img = np.clip(ref_img, rmin, rmax).astype(np.float32)

    # alignment masks
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
    ###

    fsmooth = frames.copy().astype(np.float32)

    # rigid registration
    if ops.get('norm_frames', False):
        fsmooth = np.clip(fsmooth, rmin, rmax)

    # calculate shifts
    ymax, xmax, cmax = rigid.phasecorr(
        data=rigid.apply_masks(data=fsmooth, maskMul=mask_mul, maskOffset=mask_offset),
        cfRefImg=cfRefImg,
        maxregshift=ops['maxregshift'],
        smooth_sigma_time=ops['smooth_sigma_time'],
    )
    rigidOffsets = [ymax, xmax]

    # apply shifts
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

        # calculate shifts
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

        # apply shifts
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
    create rois x Ly x Lx array of roi masks
    :param stats: suite2p stats array (iscell filter applied)
    :param ops: suite2p ops dictionary
    :return: roistack: np.array
    """

    roistack = np.zeros([stats.shape[0], ops['Ly'], ops['Lx']]).astype(np.float32)
    for i, roi in enumerate(stats):
        roistack[i, roi['ypix'], roi['xpix']] = 1
    return roistack


def get_com_from_roistack(roistack):
    """
    get center of mass for each roi in roistack


    :param roistack: np.array, [rois, Ly, Lx], array of roi masks
    :return: com: np.array [rois, 2] center of mass of each roi
    """
    com = np.zeros([roistack.shape[0], 2])
    for ind in range(roistack.shape[0]):
        com[ind, :] = np.argwhere(roistack[ind, :, :]).mean(axis=0)
    return com
