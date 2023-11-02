# import os, numpy as np, time, sys, argparse
import argparse

import numpy as np

from utils.tracking_utils import *
from AB3DMOT_libs.utils import Config, get_subfolder_seq, initialize, AB3DMOT
from easydict import EasyDict as edict
from visual_utils import draw_scenes, draw_scenes_raw  # ,Simple_Rendering_Of_Cloud_And_Boxes
from Xinshuo_PyToolbox_master.xinshuo_miscellaneous import get_timestring, print_log
from Xinshuo_PyToolbox_master.xinshuo_io import mkdir_if_missing, save_txt_file
from AB3DMOT_libs.io import load_detection, get_saving_dir, get_frame_det, save_results, save_affinity
#from AB3DMOT_libs.kitti_calib import Calibration
#import cc3d
#from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS, cluster_optics_dbscan
#from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KDTree
from AB3DMOT_libs.dist_metrics import iou_raw

from box import Box3D
import os
import yaml
import time
import scipy.spatial

input_from_SphereFormer = False  # same as from panoptic_polarnet : labels are  not 0-9 for things
with_sc = False

if with_sc:
    print("running with SC!")


def detectOverlappingBoxes2(array_of_boxes):
    def check1DOverlap(min1, max1, min2, max2):
        l = abs((max2 + min2) / 2 - (max1 + min1) / 2)
        l1 = (max1 - min1) / 2
        l2 = (max2 - min2) / 2
        gap = l - l1 - l2
        if gap >= 0.0:
            return False
        return True

    # https://gamedevelopment.tutsplus.com/collision-detection-using-the-separating-axis-theorem--gamedev-169t
    nbox = len(array_of_boxes)
    overlapping_boxes = np.array([])
    for i in range(nbox):  # range(start, stop, step)
        h, w, l, x, y, z, theta = array_of_boxes[i]
        box_a = Box3D(x=x, y=y, z=z, h=h, w=w, l=l, ry=theta)
        corners_a = Box3D.box2corners3d_camcoord(box_a)
        for j in range(i + 1, nbox):
            h, w, l, x, y, z, theta = array_of_boxes[j]
            box_b = Box3D(x=x, y=y, z=z, h=h, w=w, l=l, ry=theta)
            corners_b = Box3D.box2corners3d_camcoord(box_b)

            # draw_scenes(points=np.zeros((1,3)), ref_boxes=[box_a],
            #                gt_boxes=[box_b],
            #                title="box a vs box b")

            in_x = check1DOverlap(np.min(corners_a[:, 0]), np.max(corners_a[:, 0]),
                                  np.min(corners_b[:, 0]), np.max(corners_b[:, 0]))
            in_y = check1DOverlap(np.min(corners_a[:, 1]), np.max(corners_a[:, 1]),
                                  np.min(corners_b[:, 1]), np.max(corners_b[:, 1]))
            in_z = check1DOverlap(np.min(corners_a[:, 2]), np.max(corners_a[:, 2]),
                                  np.min(corners_b[:, 2]), np.max(corners_b[:, 2]))

            if in_x and in_y and in_z:
                overlapping_boxes = np.vstack((overlapping_boxes, np.expand_dims(np.array((i, j)), axis=0))) if len(
                    overlapping_boxes) != 0 else np.expand_dims(np.array((i, j)), axis=0)

    return overlapping_boxes


def detectOverlappingPoints(points, array_of_boxes, array_of_points_of_boxes):
    nbox = len(array_of_points_of_boxes)
    overlapping_boxes = np.array([])
    for i in range(nbox):  # range(start, stop, step)
        box_a_points_ind = array_of_points_of_boxes[i]
        box_a = array_of_boxes[i]
        for j in range(i + 1, nbox):
            box_b_points_ind = array_of_points_of_boxes[j]
            box_b = array_of_boxes[j]
            overlap_ind = np.intersect1d(box_a_points_ind, box_b_points_ind)
            if len(overlap_ind) > 3:
                """
                draw_scenes_raw(points=points[box_a_points_ind,:3], ref_boxes=[box_a],
                                gt_boxes=None,
                                title="box_a")
                draw_scenes_raw(points=points[:, :3], ref_boxes=None,
                                gt_boxes=[box_b],
                                title="box_b")
                draw_scenes_raw(points=points[overlap_ind.astype(np.int)], ref_boxes=[box_a],
                                gt_boxes=[box_b],
                                title="detectOverlappingBoxes, box_a green, box_b blue, box_a,box_b =" + str(
                                    i) + "," + str(j))
                """

                overlapping_boxes = np.vstack((overlapping_boxes, np.expand_dims(np.array((i, j)), axis=0))) if len(
                    overlapping_boxes) > 3 else np.expand_dims(np.array((i, j)), axis=0)

    non_overlapping_boxes = []
    overlap_as_list = np.unique(overlapping_boxes.ravel())
    for i in range(nbox):
        if i not in overlap_as_list:
            non_overlapping_boxes.append(i)

    return overlapping_boxes, non_overlapping_boxes


def detectOverlappingBoxes(array_of_boxes):
    nbox = len(array_of_boxes)
    overlapping_boxes = np.array([])
    for i in range(nbox):  # range(start, stop, step)
        box_a = array_of_boxes[i]
        for j in range(i + 1, nbox):
            box_b = array_of_boxes[j]
            """

            draw_scenes_raw(points= np.zeros((1,4)), ref_boxes=[box_a],
                            gt_boxes=[box_b], title="detectOverlappingBoxes, box_a green, box_b blue, box_a,box_b =" + str(i) +"," + str(j))
            """
            d = iou_raw(box_a, box_b, metric='iou_3d')
            # print('detectOverlappingBoxes : box_a ([h,w,l,x,y,z,theta])=', box_a)
            # print('detectOverlappingBoxes : box_b ([h,w,l,x,y,z,theta])=', box_b)
            # print("i,j,d = ",i," , ",j," , ",d)
            if d > 0.001:
                overlapping_boxes = np.vstack((overlapping_boxes, np.expand_dims(np.array((i, j)), axis=0))) if len(
                    overlapping_boxes) != 0 else np.expand_dims(np.array((i, j)), axis=0)
                """
                draw_scenes_raw(points=np.zeros((1, 4)), ref_boxes=[box_a],
                                gt_boxes=[box_b],
                                title="detectOverlappingBoxes, box_a green, box_b blue, box_a,box_b =" + str(
                                    i) + "," + str(j))
                """

    non_overlapping_boxes = []
    overlap_as_list = np.unique(overlapping_boxes.ravel())
    for i in range(nbox):
        if i not in overlap_as_list:
            non_overlapping_boxes.append(i)

    return overlapping_boxes, non_overlapping_boxes


def get_filesnames(parent_path, extension):
    if parent_path is not None and os.path.isdir(parent_path):
        filesnames = []
        print("reading filenames from ", parent_path)
        # populate the filenames
        for root, dirs, files in os.walk(os.path.expanduser(parent_path)):
            for file in files:
                if file.lower().endswith(extension):
                    filesnames.append(os.path.join(root, file))
        filesnames.sort()
    else:
        filesnames = None
    return filesnames


def parse_args():
    parser = argparse.ArgumentParser(description='AB3DMOT')
    parser.add_argument('--dataset', type=str,
                        default='/media/nirit/mugiwara/datasets/SemanticKitti', help='dataset path')
    parser.add_argument('--data_cfg', type=str,
                        default='/media/nirit/mugiwara/datasets/SemanticKitti/semantic-kitti.yaml',
                        help='path to config file ')
    parser.add_argument('--predictions', type=str,
                        default='./predictions_data',
                        help='path to prediction ')
    parser.add_argument('--sequences', type=int, default=8, help='sequence number ')
    parser.add_argument('--split', type=str, default='valid', help='valid or not ')
    args = parser.parse_args()
    return args


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])
def simplePCA(arr):
    '''
    # taken from https://github.com/fvilmos/simplePCA/blob/master/simplePCA.py
    :param arr: input array of shape shape[N,M]
    :return:
        mean - center of the multidimensional data,
        eigenvalues - scale,
        eigenvectors - direction
    '''

    # calculate mean
    m = np.mean(arr, axis=0)

    # center data
    arrm = arr - m

    # calculate the covariance, decompose eigenvectors and eigenvalues
    # M * vect = eigenval * vect
    # cov = M*M.T
    Cov = np.cov(arrm.T)
    eigval, eigvect = np.linalg.eig(
        Cov.T)  # column eigvect[:,i] is the eigenvector corresponding to the eigenvalue eigval[i].

    # return mean, eigenvalues, eigenvectors
    return m, eigval, eigvect


"""
def rotm2axang(rotm):
    # https://github.com/robotology-legacy/mex-wholebodymodel/blob/master/mex-wholebodymodel/matlab/utilities/%2BWBM/%2Butilities/rotm2axang.m
    # Translate a given rotation matrix R into the corresponding axis-angle representation (u, theta).
    # For further details about the computation, see:
    #   [1] Technical Concepts: Orientation, Rotation, Velocity and Acceleration and the SRM, P. Berner, Version 2.0, 2008,
    #       <http://sedris.org/wg8home/Documents/WG80485.pdf>, pp. 32-33.
    #   [2] A Mathematical Introduction to Robotic Manipulation, Murray & Li & Sastry, CRC Press, 1994, p. 30, eq. (2.17) & (2.18).
    #   [3] Modelling and Control of Robot Manipulators, L. Sciavicco & B. Siciliano, 2nd Edition, Springer, 2008,
    #       p. 35, formula (2.25).
    #   [4] Introduction to Robotics: Mechanics and Control, John J. Craig, 3rd Edition, Pearson/Prentice Hall, 2005,
    #       pp. 47-48, eq. (2.81) & (2.82).

    if rotm.shape!=(3,3):
        raise RuntimeError("rotm SHOULD BE 3X3")

    axang   = np.zeros((4,1))
    epsilon = 1e-12 # min. value to treat a number as zero ...

    tr = rotm[0, 0] + rotm[1, 1] + rotm[2,2]
    if abs(tr - 3) <= epsilon
        #  Null rotation --> singularity: The rotation matrix R is the identity matrix and the axis of rotation u is undefined.
        #  By convention, set u to the default value (0, 0, 1) according to the ISO/IEC IS 19775-1:2013 standard of the Web3D Consortium.
        #    See: <http://www.web3d.org/documents/specifications/19775-1/V3.3/Part01/fieldsDef.html#SFRotationAndMFRotation>
        axang[2, 0] = 1
    elif (abs(tr + 1) <= epsilon):# tr = -1 --> theta = pi:
         if (rotm[0,0] > rotm[1,1]) and (rotm[0,0] > rotm[2,2]):
             u = np.vstack(rotm[0, 0] + 1, rotm[0, 1], rotm[0, 2])
         elif (rotm[1,1] > rotm[2,2]):
             u = np.vstack(rotm[1, 0], rotm[1, 1] + 1, rotm[1, 2])
         else:
             u = np.vstack(rotm[2, 0], rotm[2, 1], rotm[2, 2] + 1)
         n = np.matmul(u.T,u)
         axang[0:2, 1] = np.divide(u,np.sqrt(n)) # 0,1,2 should be filled with new vvalues
         axang[3, 0] = np.pi
    else:
        # general case, tr ~= 3 and tr ~= -1:
        axang[3, 0] = np.arccos((tr - 1) * 0.5)
        n_inv = 1/(2*np.sin(axang[3,0]))
        # % unit vector u:
        axang[0, 0] = (rotm[2, 1] - rotm[1, 2]) * n_inv
        axang[1, 0] = (rotm[0, 2] - rotm[2, 0]) * n_inv
        axang[2, 0] = (rotm[1, 0] - rotm[0, 1]) * n_inv
    return axang
"""

"""
def rotation_angles(matrix, order='xyz'):
    #https://programming-surgeon.com/en/euler-angle-python-en/

    #input
    #    matrix = 3x3 rotation matrix (numpy array)
    #    oreder(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
    #output
    #    theta1, theta2, theta3 = rotation angles in rotation order

    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    if order == 'xzx':
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == 'xyx':
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 *np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == 'yxy':
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == 'yzy':
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == 'zyz':
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == 'zxz':
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == 'xzy':
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == 'xyz':
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == 'yxz':
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == 'yzx':
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == 'zyx':
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == 'zxy':
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)

    # to degree
    #theta1 = theta1 * 180 / np.pi
    #theta2 = theta2 * 180 / np.pi
    #theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)
"""


def save_detection(filename, dets_frame_vehicles, dets_frame_bikes, dets_frame_pedestrian, det_id2str, frameNum):
    file = open(filename, 'w')
    save_detection_local(file, dets_frame_vehicles, det_id2str, frameNum)
    save_detection_local(file, dets_frame_bikes, det_id2str, frameNum)
    save_detection_local(file, dets_frame_pedestrian, det_id2str, frameNum)
    file.close()


def save_detection_local(file, dets, det_id2str, frameNum):
    for j in range(len(dets['dets'])):
        h, w, l, x, y, z, theta = dets['dets'][j]
        classID, TrackID, score = dets['info'][j][1], dets['info'][j][2], dets['info'][j][6]
        # expecting  format [bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry](1x7) ,[trk.id](1x1), trk.info[1x8])

        result_tmp = np.array(
            (h, w, l, x, y, z, theta, TrackID, np.nan, classID, np.nan, np.nan, np.nan, np.nan, score))

        save_results(result_tmp, file, None, det_id2str, frameNum, -1000000)


# Oriantation
def points_to_Box3D2(points, class_label=None):
    #:param points: instance points Nx3
    #:return: 3D bbox # [cx, cy, cz, theta, l, w, h]

    # draw_scenes(points, gt_boxes=None, ref_boxes=None, title='points_to_Box3D: input points')
    # idea from
    # https://stackoverflow.com/questions/58632469/how-to-find-the-orientation-of-an-object-shape-python-opencv
    # https://towardsdatascience.com/visualizing-principal-component-analysis-with-matrix-transforms-d17dabc8230e
    # https://stackoverflow.com/questions/28701213/how-to-find-principal-components-and-orientation-of-a-point-cloud-using-point-cl
    # 1.  compute the orientation
    # return mean, eigenvalues, eigenvectors
    mean, eigenvalues, eigenvectors = simplePCA(points)

    sort_indices = np.argsort(eigenvalues)[::-1]
    x_v1, y_v1, z_v1 = eigenvectors[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2, z_v2 = eigenvectors[:, sort_indices[1]]
    x_v3, y_v3, z_v3 = eigenvectors[:, sort_indices[2]]  # Eigenvector with smallest eigenvalue

    # from the five rules
    # https://bioturing.medium.com/the-why-when-and-how-of-3d-pca-bdb5c209f693
    # print('eigenvalues =',eigenvalues[sort_indices])
    if eigenvalues[sort_indices][0] >= 1.0:
        ry = - np.arctan2(x_v1, y_v1)
    else:
        ry = 0  # PCA cannot determine the orientation

    """
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    scale = 1
    plt.plot([x_v1 * -scale * 2, x_v1 * scale * 2],
             [y_v1 * -scale * 2, y_v1 * scale * 2],
             [z_v1 * -scale * 2, z_v1 * scale * 2],color='red')
    plt.plot([x_v2 * -scale, x_v2 * scale],
             [y_v2 * -scale, y_v2 * scale],
             [z_v2 * -scale, z_v2 * scale],color='blue')
    plt.plot([x_v3 * -scale, x_v3 * scale],
             [y_v3 * -scale, y_v3 * scale],
             [z_v3 * -scale, z_v3 * scale],color='green')
    ax.scatter3D(points[:,0],
             points[:,1],
             points[:,2],
                 'black')
    #plt.axis('equal')
    #plt.gca().invert_yaxis()  # Match the image system with origin at top left
    plt.show()
    """

    # https://stackoverflow.com/questions/28701213/how-to-find-principal-components-and-orientation-of-a-point-cloud-using-point-cl

    # eigenvectors
    R = roty(ry)

    # a,b,c=rotation_angles(R, order='zxy')

    # yaw,pitch,roll= rotation_angles(pose[:3,:3], order='zxy')

    # print(" yaw,pitch,roll",  yaw,pitch,roll)

    # print("simplePCA :  a,b,c", a,b,c)
    # https://towardsdatascience.com/orientation-estimation-in-monocular-3d-object-detection-f850ace91411
    # ry = -yaw - a#+ a #0 #a +np.pi/2  #a -np.pi/2

    # import transforms3d.euler as t3d
    # rot = t3d.quat2mat([poses[i, 3], poses[i, 4], poses[i, 5], poses[i, 6]])
    # euler_cam = t3d.mat2euler(pose[:3,:3], 'szxy')
    # euler_local = t3d.mat2euler(R, 'szxy')
    # print("euler_local: yaw,pitch,roll", euler_local[0], euler_local[1], euler_local[2])
    # print("euler_cam: yaw,pitch,roll", euler_cam[0], euler_cam[1], euler_cam[2])
    # ry = euler_local[1]#eul                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              er_local[2] #-(euler_cam[1]-euler_local[1])

    # a= rotm2axang(R.T)
    # print("rotm2axang=",a)
    # ry = a[3]+np.pi/2

    # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Conversion_formulae_between_formalisms
    # yaw = np.arctan2(R(2, 1), R(1, 1))
    # atan2(R[1][0] / cos(pitch), R[0][0] / cos(pitch));

    # An intuition of why the large eigenvalue corresponds to the short axis.
    # https://math.stackexchange.com/questions/1447730/drawing-ellipse-from-eigenvalue-eigenvector

    # main_axes = eigve[:, np.argmin(eigva)]

    # orientation-estimation (did not use this at th end):
    #  https://towardsdatascience.com/orientation-estimation-in-monocular-3d-object-detection-f850ace91411
    #  https://arxiv.org/pdf/1612.00496.pdf

    ## debug result
    # scale the eienvector and reduce with half
    # import matplotlib
    # matplotlib.use('Qt5Agg')  # https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so
    # import matplotlib.pyplot as plt
    # from mpl_toolkits import mplot3d
    # scale the eienvector and reduce with half

    # fig = plt.figure(figsize=(4, 4))
    # ax = mplot3d.Axes3D(fig)
    # ax.scatter3D(points.T[0], points.T[1], points.T[2], c='r')
    # ax.quiver(m[0], m[1], m[2], eigve[:, 0], eigve[:, 1], eigve[:, 2],
    #          color=[(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 1), (0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0),
    #                 (1, 0, 0)],  ## line1, line2, line3, l1arrow1, l1arrow2, l2arrow1, l2arrow2, l3arrow1, l3arrow2
    #          length=0.5, normalize=True)
    # ax.plot3D([m[0],main_axes[0]], [m[1],main_axes[1]], [m[2],main_axes[2]], c='black')
    ## ax.scatter3D(main_axes[0]*0.5*np.min(eigva), main_axes[1]*0.5*np.min(eigva) ,main_axes[2]*0.5*np.min(eigva), c='black')

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    # ax.view_init(azim=30)
    # plt.show()

    # draw_scenes(points, gt_boxes=None, ref_boxes=None, title='input points_to_Box3D')
    # https://logicatcore.github.io/scratchpad/lidar/sensor-fusion/jupyter/2021/04/20/3D-Oriented-Bounding-Box.html

    centered_data = points - mean
    aligned_coords = np.matmul(R.T, centered_data.T).T

    x_min = np.min(aligned_coords[:, 0])
    x_max = np.max(aligned_coords[:, 0])
    y_min = np.min(aligned_coords[:, 1])
    y_max = np.max(aligned_coords[:, 1])
    z_min = np.min(aligned_coords[:, 2])
    z_max = np.max(aligned_coords[:, 2])

    # realigned_coords = np.matmul(R, aligned_coords.T).T
    # realigned_coords += mean

    l = x_max - x_min  # x
    w = y_max - y_min  # y
    h = z_max - z_min  # z

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    # x, y, z, theta, l, w, h
    cx = x_min + (x_max - x_min) / 2
    cy = y_min + (y_max - y_min) / 2
    cz = z_min + (z_max - z_min) / 2

    bb = Box3D(x=cx, y=cy, z=cz, h=h, w=w, l=l, ry=ry, c=class_label)

    # draw_scenes(points, gt_boxes=None, ref_boxes=[bb],title='final points_to_Box3D')

    return bb


# normal
def points_to_Box3D(points, class_label=None):
    #:param points: instance points Nx3
    #:return: 3D bbox # [cx, cy, cz, theta, l, w, h]
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    # x, y, z, theta, l, w, h
    cx = x_min + (x_max - x_min) / 2
    cy = y_min + (y_max - y_min) / 2
    cz = z_min + (z_max - z_min) / 2

    l = x_max - x_min
    w = y_max - y_min
    h = z_max - z_min

    if l < 0 or w < 0 or h < 0:
        print('points_to_Box3D :  error')

    ry = 0

    bb = Box3D(x=cx, y=cy, z=cz, h=h, w=w, l=l, ry=ry, c=class_label)

    # draw_scenes(points, gt_boxes=None, ref_boxes=[bb],title='debug points_to_Box3D')
    return bb


# using instances
def get_detections(points, label_inst, label_sem_class, scores, point_grid=None, grid_sem=None, minPoints=None):
    ins_ids = np.unique(label_inst)

    # get instances from current frames to track
    bb_list = []
    additional_info = []
    # fill info array
    info_array = np.empty([1, 7])
    info_array[:] = np.nan
    sc_items = []
    if grid_sem is not None:
        grid_interesting_idx = np.where(grid_sem != 0)[0]

    all_obj_points = []
    for ins_id in ins_ids:
        if ins_id == 0:
            continue
        obj_loc = np.where(label_inst == ins_id)[0]
        obj_points = points[obj_loc, :]
        obj_points = obj_points[:, :3]
        object_scores = scores[obj_loc]

        # draw_scenes(points=points, title='get_detections: all points')

        # draw_scenes(points=point_grid[grid_intersting_idx], title='get_detections: point_grid')
        # draw_scenes(points=obj_points, title='get_detections: obj_points')
        # draw_scenes(points=np.vstack((obj_points,point_grid[grid_intersting_idx])), title='get_detections: obj_points  with grid points')
        obj_with_sc = False
        if grid_sem is not None:
            if len(grid_interesting_idx) != 0:
                # to avoid calc kd-tree if obj_points is not in range of sc:
                x_min = np.min(obj_points[:, 0])
                x_max = np.max(obj_points[:, 0])
                y_min = np.min(obj_points[:, 1])
                y_max = np.max(obj_points[:, 1])
                z_min = np.min(obj_points[:, 2])
                z_max = np.max(obj_points[:, 2])

                # JS3C range:
                min_extent = [0, -25.6, -2]
                max_extent = [51.2, 25.6, 4.4]

                if ((x_min <= min_extent[0] <= x_max) or (x_min <= max_extent[0] <= x_max) or (
                        x_min >= min_extent[0] and x_max <= max_extent[0])) and \
                        ((y_min <= min_extent[1] <= y_max) or (y_min <= max_extent[1] <= y_max) or (
                                y_min >= min_extent[1] and y_max <= max_extent[1])):

                    # start = time.time()

                    # draw_scenes(points=obj_points, title='get_detections: input :  obj_points')

                    tree = KDTree(point_grid[grid_interesting_idx], leaf_size=2)
                    dist, ind = tree.query(obj_points, k=int(np.sqrt(len(grid_interesting_idx))))
                    f_ind = []
                    dist_all = []
                    ind_all = []
                    for ro, d in enumerate(dist):
                        for co, dd in enumerate(d):
                            dist_all.append(d)
                            ind_all.append(ind[ro, co])
                            if dd < 1.5:
                                f_ind.append(ind[ro, co])
                    f_ind = np.unique(f_ind)
                    if len(f_ind) >= minPoints:
                        obj_points = np.vstack((obj_points, point_grid[grid_interesting_idx[f_ind]]))
                        obj_with_sc = True
                        # draw_scenes(points=obj_points, title='get_detections: first iter, obj_points  with grid points')

                        grid_interesting_idx = np.delete(grid_interesting_idx, f_ind)
                    # else:
                    #    if len(f_ind)==0:
                    #        # probably miss detection
                    #        obj_points =[]
                    # end = time.time() - start
                    # print("get_detections : time = ",end)
            else:
                # draw_scenes(points=obj_points, title='get_detections: first iter, obj_points  with grid points')
                print("empty grid_interesting_idx, debug here!")

        #  filter too small area detection
        if len(obj_points) < 2:
            continue

        obj_class_label = None
        if len(np.unique(label_sem_class[obj_loc])) != 1:
            print('ops... something went wrong in box_tracker.py ')
            print('  ins_id = ', ins_id)
            print('  np.unique(label_sem_class[obj_loc] = ', np.unique(label_sem_class[obj_loc]))

            # I reached here with GT data of seq8 frame0
            raise RuntimeError("get_detections: something went wrong!")
            values, counts = np.unique(label_sem_class[obj_loc], return_counts=True)
            ind = np.argmax(counts)
            obj_class_label = values[ind]
        else:
            obj_class_label = label_sem_class[obj_loc[0]]

        # bb = points_to_Box3D(obj_points, obj_class_label)
        # bb = points_to_Box3D2(obj_points) # orientation try
        bb = points_to_Box3D(obj_points)
        info_array[0, 1] = obj_class_label
        info_array[0, 2] = ins_id
        info_array[0, 6] = np.max(object_scores)

        # filter box  with  w/h/l smaller than 0.2 (20cm)
        # nirit - another protection, still havn't seen a case on which this is true
        if \
                bb.l < 0.2 or \
                        bb.w < 0.2 or \
                        bb.h < 0.2:
            # draw_scenes_raw(points=points, gt_boxes=np.expand_dims(Box3D.bbox2array_raw(bb), 0), title='get_detections : to remove small detections')
            continue

        if \
                bb.l > 30.0 or \
                        bb.w > 30.0 or \
                        bb.h > 30.0:
            # print('  filtered box with : bb.l ,bb.w, bb.h, class = ', bb.l, bb.w, bb.h,cls_id)
            # draw_scenes_raw(points=points, gt_boxes=np.expand_dims(Box3D.bbox2array_raw(bb), 0),
            #                title='get_detections : to remove large detections')
            continue

        # bb_list.append(bb)
        bb_raw = np.expand_dims(Box3D.bbox2array_raw(bb), 0)  # [[h,w,l,x,y,z,theta],...]
        bb_list = np.vstack((bb_list, bb_raw)) if len(bb_list) != 0 else bb_raw
        # all_obj_points = np.vstack((all_obj_points, obj_points)) if len(all_obj_points) != 0 else obj_points
        all_obj_points.append(obj_points)

        additional_info = np.vstack((additional_info, info_array)) if len(additional_info) != 0 else info_array.copy()

        sc_items.append(obj_with_sc)

    # draw_scenes_raw(points=points, gt_boxes=[Box3D.bbox2array_raw(bb)], title='get_detections : detections')
    """

    sc_all_points =[]
    results = []
    start_ind = 0
    for ind,l in enumerate(all_obj_points):
        if sc_items[ind]:
            results.append([i for i in range(start_ind, len(l) + start_ind)])
            #results.append([ i for i in range(start_ind, len(l) + start_ind)])
        sc_all_points = np.vstack((sc_all_points, l)) if len(sc_all_points) != 0 else l
        start_ind = len(sc_all_points) + 1

    #if len(sc_all_points)==0:
    #    draw_scenes_raw(points=points, gt_boxes=dets_frame['dets'], title='get_detections : all detections')
    #else:
    #    draw_scenes_raw(points=sc_all_points, gt_boxes=dets_frame['dets'], title='get_detections : all detections')    

    """
    dets_frame = {'dets': bb_list, 'info': additional_info}
    # draw_scenes_raw(points=points, gt_boxes=dets_frame['dets'], title='get_detections : all detections')
    """
    #	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
    #	info: a array of other info for each det



    #draw_scenes_raw(points=sc_all_points, gt_boxes=dets_frame['dets'], title='get_detections : all detections')

    sc_ind= np.where(sc_items)[0]
    if len(sc_ind)!=0:
        sc_boxes = bb_list[sc_ind,:]
        sc_info  = additional_info[sc_ind,:]

        #overlapping_boxes, non_overlapping_boxes = detectOverlappingPoints(sc_all_points, sc_boxes, results)  # there is overlap if points overlap
        #draw_scenes_raw(points=sc_all_points, gt_boxes=sc_boxes, title='get_detections : sc boxes')

        #s= time.time()
        overlapping_boxes, non_overlapping_boxes = detectOverlappingBoxes(sc_boxes)
        #print("detectOverlappingBoxes : ", time.time()-s)

        #s = time.time()
        #overlapping_boxes = detectOverlappingBoxes2(sc_boxes)
        #print("detectOverlappingBoxes2 : ", time.time() - s)

        #s = time.time()
        merge_bb = []
        merge_info = []
        idx_to_delete = []
        for i in range(len(overlapping_boxes)):
            overlapping_box0 = overlapping_boxes[i, 0]
            overlapping_box1 = overlapping_boxes[i, 1]
            union_obj_points = np.vstack((all_obj_points[sc_ind[overlapping_box0]],all_obj_points[sc_ind[overlapping_box1]]))
            #np.union1d( , all_obj_points[sc_ind[overlapping_box1]])

            bb = points_to_Box3D(union_obj_points)
            info_array[0, 1] = sc_info[overlapping_box0, 1] if len(results[overlapping_box0]) >= len(results[overlapping_box1]) else sc_info[overlapping_box1, 1]
            info_array[0, 2] = sc_info[overlapping_box0, 2] if len(results[overlapping_box0]) >= len(results[overlapping_box1]) else sc_info[overlapping_box1, 2]
            info_array[0, 6] = np.max([sc_info[overlapping_box0,6],sc_info[overlapping_box1,6]])

            if \
                    bb.l < 0.2  or\
                    bb.w < 0.2  or\
                    bb.h < 0.2:
                #draw_scenes_raw(points=points, gt_boxes=np.expand_dims(Box3D.bbox2array_raw(bb), 0), title='get_detections : to remove small detections')
                continue

            if \
                    bb.l > 30.0 or \
                            bb.w > 30.0 or \
                            bb.h > 30.0:
                # print('  filtered box with : bb.l ,bb.w, bb.h, class = ', bb.l, bb.w, bb.h,cls_id)
                #draw_scenes_raw(points=points, gt_boxes=np.expand_dims(Box3D.bbox2array_raw(bb), 0),
                #                title='get_detections : to remove large detections')
                continue


            bb_raw = np.expand_dims(Box3D.bbox2array_raw(bb), 0)  # [[h,w,l,x,y,z,theta],...]
            merge_bb = np.vstack((merge_bb, bb_raw)) if len(merge_bb) != 0 else bb_raw
            merge_info = np.vstack((merge_info, info_array)) if len(merge_info) != 0 else info_array.copy()

            idx_to_delete.append(sc_ind[overlapping_box0])
            idx_to_delete.append(sc_ind[overlapping_box1])

        # remove overlapping
        if len(idx_to_delete) !=0:
            bb_list_new =  np.array([l for ind,l in enumerate(bb_list) if ind not in idx_to_delete ])
            additional_info_new = np.array([l for ind, l in enumerate(additional_info) if ind not in idx_to_delete])
            # add merged boxes
            bb_list_new = np.vstack((bb_list_new, merge_bb)) if len(bb_list_new) != 0 else merge_bb
            additional_info_new = np.vstack((additional_info_new, merge_info)) if len(additional_info_new) != 0 else merge_info.copy()
        else:
            bb_list_new = bb_list
            additional_info_new = additional_info
        #print("merge loop  : ", time.time() - s)
        dets_frame = {'dets': bb_list_new, 'info': additional_info_new}
    #	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
    #	info: a array of other info for each det
    #draw_scenes_raw(points=sc_all_points, gt_boxes=dets_frame['dets'],title='get_detections : all detections')
    """
    return dets_frame


def filterPedestrian(dets_all):
    dets, info = dets_all['dets'], dets_all['info']

    idx_to_keep = []
    for d, det in enumerate(dets):
        h, w, l, x, y, z, theta = det

        if l <= 1.6 and w <= 1.6:
            idx_to_keep.append(d)
    if len(idx_to_keep) == 0:
        return dets_all
    return {'dets': dets[idx_to_keep], 'info': info[idx_to_keep, :]}


def get_detections_from_segmentation(points, label_sem_class):
    class_ids = np.unique(label_sem_class)

    # get instances from current frames to track
    bb_list = []
    additional_info = []
    # fill info array
    info_array = np.empty([7])
    info_array[:] = np.nan
    for cls_id in class_ids:
        if cls_id == 0:
            continue
        obj_loc = np.where(label_sem_class == cls_id)[0]
        obj_points = points[obj_loc, :3]
        # print('class id = ', cls_id, ': size = ', len(obj_loc))

        # draw_scenes(points=obj_points, title='get_detections_from_segmentation: obj_points')

        #  filter too small area detection
        # box should contain 2 points and more
        if len(obj_loc) < 2:
            continue

        # draw_scenes(points=obj_points, title='get_detections_from_segmentation: obj_points')

        # clustering = DBSCAN(eps=2, min_samples=4).fit(obj_points)
        # labels_results = clustering.labels_

        clust = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
        clust.fit(obj_points)
        labels_results = clust.labels_

        # print ('np.unique(labels_results) =',np.unique(labels_results))

        for c_lbl in np.unique(labels_results):

            if c_lbl == -1:  # -1 is noisy sample. ignore it.
                continue

            obj = obj_points[np.where(labels_results == c_lbl)[0], :]
            bb = points_to_Box3D(obj)

            info_array[1] = cls_id

            # print('box with : bb.l ,bb.w, bb.h, class = ', bb.l, bb.w, bb.h, cls_id)

            # filter out small boxes
            # filter box  with zero w/h/l
            # nirit - another protection, still havn't seen a case on which this is true
            if \
                    bb.l < np.finfo(float).eps or \
                            bb.w < np.finfo(float).eps or \
                            bb.h < np.finfo(float).eps:
                continue

            # filter out big boxes # bus  max size is 18 meter , so 20  was chosen
            if \
                    bb.l > 30.0 or \
                            bb.w > 30.0 or \
                            bb.h > 30.0:
                # print('  filtered box with : bb.l ,bb.w, bb.h, class = ', bb.l, bb.w, bb.h,cls_id)
                continue

            # bb_list.append(bb)
            bb_raw = Box3D.bbox2array_raw(bb)  # [[h,w,l,x,y,z,theta],...]
            bb_list = np.vstack((bb_list, bb_raw)) if len(bb_list) != 0 else np.array(bb_raw, ndmin=2)
            additional_info = np.vstack((additional_info, info_array)) if len(additional_info) != 0 else np.array(
                info_array, ndmin=2)

        # draw_scenes(points=obj_points, title='get_detections_from_segmentation: obj_points')
        # draw_scenes_raw(points=points, gt_boxes=bb_list, title='get_detections_from_segmentation : detections')

        """ did not work, got some error
        mask3D = np.zeros((len(obj_loc),4),dtype=bool)
        mask3D[:3,:] = True
        mask3D[obj_loc, :] = obj_loc


        draw_scenes(points = obj_points,title='obj_points')

        connectivity = 6  # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
        #labels_out= cc3d.connected_components(mask3D, connectivity=connectivity)
        labels_out,N = cc3d.connected_components(mask3D, connectivity=connectivity,return_N=True)


        #from cloudvolume import view
        #view(labels_out, segmentation=True)

        # Image statistics like voxel counts, bounding boxes, and centroids.
        stats = cc3d.statistics(labels_out)

        for segid in range(1, N + 1):
            extracted_image = obj_points * (labels_out == segid)
            draw_scenes(points=extracted_image, title='extracted_image')
            process(extracted_image)  # stand in for whatever you'd like to do


        for label, image in cc3d.each(labels_out, binary=False, in_place=True):
            loc = np.where(labels_out == label)[0]
            bb = points_to_Box3D(points[loc, :3])


            print('hi')
            #process(image)  # stand in for whatever you'd like to do


            bb = points_to_Box3D(obj_points)
            info_array[1] = obj_class_label
            # filter box  with zero w/h/l
            # nirit - another protection, still havn't seen a case on which this is true
            if \
                    bb.l <  np.finfo(float).eps or\
                    bb.w <  np.finfo(float).eps or\
                    bb.h <  np.finfo(float).eps:
                continue

            #bb_list.append(bb)
            bb_list = np.vstack((bb_list, Box3D.bbox2array_raw(bb))) if len(bb_list) != 0 else Box3D.bbox2array_raw(bb) # [[h,w,l,x,y,z,theta],...]
            additional_info = np.vstack((additional_info, info_array)) if len(additional_info)!=0  else info_array
        """

    # draw_scenes_raw(points=points, gt_boxes=bb_list, title='detections')
    dets_frame = {'dets': bb_list, 'info': additional_info}
    #	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
    #	info: a array of other info for each det

    return dets_frame


def remove_detections(dets_frame, idx):
    def findCloset(dets_frame, cx, cy, cz):

        float_eps = 0.5  # 0.1#np.finfo(np.float64).eps
        for j in range(len(dets_frame['dets'])):
            h, w, l, x, y, z, theta = dets_frame['dets'][j]
            if (abs(cx - x) < float_eps and abs(cy - y) < float_eps and abs(cz - z) < float_eps):
                return j
        return -1

    if idx == 0:
        cx, cy, cz = 17.319663, 4.205831, -0.825820
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)

    """
    # for seq 08  gt id4,  "moving-car"
    if idx ==30:
        cx,cy,cz = 17.319663, 4.205831, -0.825820
        closeIdx =  findCloset(dets_frame,cx,cy,cz)
        if closeIdx!=-1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx,0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx,0)
    if idx  ==31:
        cx, cy, cz = 16.183018, 4.146237, -0.870163
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==32:
        cx, cy, cz = 15.061197, 4.116314, -0.895400
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==33:
        cx, cy, cz = 13.913082, 4.051011, -0.930553
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx==34:
        cx, cy, cz = 12.850779, 3.975125, -1.008494
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx==35:
        cx, cy, cz = 11.782966 ,3.896697 ,-0.976497
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx==36:
        cx, cy, cz = 10.671791 ,3.774049, -1.053867
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx==37:
        cx, cy, cz = 9.580346, 3.740621, -1.039194
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 38:
        cx, cy, cz = 8.450035 ,3.605769 ,-1.014445
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 39:
        cx, cy, cz = 7.363937, 3.450792, -1.028434
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==40:
        cx, cy, cz = 6.289426, 3.368453, -0.968179
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==41:
        cx, cy, cz = 5.171908, 3.251830 ,-1.024207
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==42:
        cx, cy, cz = 4.150034, 3.174985, -1.010644
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==43:
        cx, cy, cz = 3.077972, 3.049704 ,-1.010604
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==44:
        cx, cy, cz = 2.149396, 3.109197 ,-1.005154
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==45:
        cx, cy, cz = 1.011949, 2.985584 ,-0.90111
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 46:
        cx, cy, cz = -0.042927, 2.978626, -0.863267
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 47:
        cx, cy, cz = -1.124612, 3.035225, -0.987751
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 48:
        cx, cy, cz = -2.134746, 3.035252, -1.102249
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 49:
        cx, cy, cz = -3.168520 ,3.082112, -1.089942
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 50:
        cx, cy, cz = -4.366337, 3.205693, -0.891063
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==51:
        cx, cy, cz = -5.400810 ,3.357855, -0.938319
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==52:
        cx, cy, cz = -6.521146, 3.667977 ,-0.881207
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==53:
        cx, cy, cz = -7.617999 ,4.026076, -1.106839
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==54:
        cx, cy, cz = -8.741103 ,4.518119, -1.210565
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx ==55:
        cx, cy, cz = -9.741767, 5.088352, -1.168456
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 56:
        cx, cy, cz = -10.810380 ,5.774496, -1.271068
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 57:
        cx, cy, cz = -11.788192 ,6.695596, -1.370420
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 58:
        cx, cy, cz = -12.618711, 7.542192, -1.393519
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 59:
        cx, cy, cz = -13.458164 ,8.444260, -1.492111
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 60:
        cx, cy, cz = -14.264443 ,10.022214 ,-1.570307
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 61:
        cx, cy, cz = -14.772139, 10.665061, -1.585876,
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 62:
        cx, cy, cz = -15.396466, 12.580526 ,-1.644223
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 63:
        cx, cy, cz = -15.897800, 13.926274, -1.623961
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 64:
        cx, cy, cz = -15.975636, 15.670546, -1.683990,
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 65:
        cx, cy, cz = -16.246900, 17.026385, -1.724939
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 66:
        cx, cy, cz = -16.837837, 18.721453, -1.789247
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 67:
        cx, cy, cz = -16.496333 ,20.205410 ,-1.863987
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 68:
        cx, cy, cz = -16.425657, 21.920581 ,-1.954724
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 69:
        cx, cy, cz = -16.181338 ,23.579254 ,-2.055460
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    if idx == 70:
        cx, cy, cz = -15.407494 ,25.374043 ,-1.946691
        closeIdx = findCloset(dets_frame, cx, cy, cz)
        if closeIdx != -1:
            dets_frame['dets'] = np.delete(dets_frame['dets'], closeIdx, 0)
            dets_frame['info'] = np.delete(dets_frame['info'], closeIdx, 0)
    """

    return dets_frame


def main(args):
    # parameters from args
    prediction_dir = args.predictions
    split = args.split
    dataset = args.dataset

    if split == 'valid':
        prediction_path = '{}/val_probs'.format(prediction_dir)
    else:
        prediction_path = '{}/probs'.format(prediction_dir)
    print("input is taken from : ", prediction_path)
    base_save_path = '{}/NextStop_tracker'.format(prediction_dir)
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    full_save_path = (os.path.join(base_save_path, 'sequences'))
    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)

    config_data = yaml.safe_load(open(args.data_cfg, 'r'))

    # if input of detection comes from 4Dstop  then class are in the learning_map_inv
    if input_from_SphereFormer:
        det_id2str = config_data['labels']
    else:
        det_id2str = {key: config_data['labels'][value] for key, value in config_data['learning_map_inv'].items()}

    # get test set
    test_sequences = [args.sequences]

    poses = []
    total_time = 0.0
    for sequence in test_sequences:
        calib = parse_calibration(os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "calib.txt"))
        poses_f64 = parse_poses(os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "poses.txt"), calib)
        poses.append([pose.astype(np.float32) for pose in poses_f64])

    for poses_seq, sequence in zip(poses, test_sequences):
        point_names = []
        point_paths = os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "velodyne")
        # populate the label names
        seq_point_names = sorted(
            [os.path.join(point_paths, fn) for fn in os.listdir(point_paths) if fn.endswith(".bin")])

        point_names.extend(seq_point_names)

        # pred_voxel_path = os.path.join(pth_js3c_output, "sequences",'{0:02d}'.format(sequence), "predictions")
        if with_sc:
            pred_voxel_path = os.path.join(pth_js3c_output, "sequences", '{0:02d}'.format(sequence))
            pred_voxel_names = get_filesnames(parent_path=pred_voxel_path, extension='voxels.npy')
            pred_voxel_sem_names = get_filesnames(parent_path=pred_voxel_path, extension='sem_label.npy')

        # init save path
        parent_save_path= (os.path.join(full_save_path, '{0:02d}'.format(sequence)))
        if not os.path.exists(parent_save_path):
            os.makedirs(parent_save_path)


        full_save_path = (os.path.join(parent_save_path, 'predictions'))
        if not os.path.exists(full_save_path):
            os.makedirs(full_save_path)

        #vis_dir = os.path.join(full_save_path, 'vis_debug')
        #if not os.path.exists(vis_dir):
        #    os.makedirs(vis_dir)

        #vis_dir = os.path.join(full_save_path, 'vis_debug')
        #if not os.path.exists(vis_dir):
        #    os.makedirs(vis_dir)

        detection_save_path = (os.path.join(parent_save_path, 'detection'))
        if not os.path.exists(detection_save_path):
            os.makedirs(detection_save_path)

        # initialize tracker

        AB3DMOT_cgf = edict()
        # ------------------- General Options -------------------------
        AB3DMOT_cgf['description'] = 'AB3DMOT'
        AB3DMOT_cgf['seed'] = 0
        # --------------- main.py
        AB3DMOT_cgf['save_root'] = full_save_path
        AB3DMOT_cgf['dataset'] = 'SemanticKITTI'  # 'KITTI'
        AB3DMOT_cgf['split'] = 'val'
        AB3DMOT_cgf['det_name'] = '4D-STop'  # 'pointrcnn'
        AB3DMOT_cgf['cat_list'] = config_data['labels'].values()  # ['Car', 'Pedestrian', 'Cyclist']
        AB3DMOT_cgf['score_threshold'] = -10000
        AB3DMOT_cgf['num_hypo'] = 1
        # --------------- model.py
        AB3DMOT_cgf['ego_com'] = False  # turn on only slightly reduce speed but increase a lot for performance
        AB3DMOT_cgf['vis'] = False  # only for debug or visualization purpose, will significantly reduce speed
        AB3DMOT_cgf['affi_pro'] = True

        """
        tracker, frame_list = initialize(,
                                         data_root=trk_root,
                                         save_dir=full_save_path,
                                         subfolder,
                                         seq_name,
                                         cat,
                                         ID_start,
                                         hw,
                                         log)
        """
        # create folders for saving
        # create eval dir for each hypothesis
        eval_dir_dict = dict()
        for index in range(AB3DMOT_cgf.num_hypo):
            eval_dir_dict[index] = os.path.join(full_save_path, 'data_%d' % index);
            mkdir_if_missing(eval_dir_dict[index])

        #eval_file_dict, save_trk_dir, affinity_dir, affinity_vis = get_saving_dir(eval_dir_dict=eval_dir_dict,
        #                                                                          seq_name="sequences" + '{0:02d}'.format(
        #                                                                              sequence),
        #                                                                          save_dir=full_save_path,
        #                                                                          num_hypo=AB3DMOT_cgf.num_hypo)

        eval_file_dict, save_trk_dir = get_saving_dir(eval_dir_dict=eval_dir_dict,
                                                      seq_name="sequences" + '{0:02d}'.format(
                                                          sequence),
                                                      save_dir=full_save_path,
                                                      num_hypo=AB3DMOT_cgf.num_hypo)

        time_str = get_timestring()
        log_path = os.path.join(AB3DMOT_cgf.save_root,
                                'log/log_%s_%s_%s.txt' % (time_str, AB3DMOT_cgf.dataset, AB3DMOT_cgf.split))
        mkdir_if_missing(log_path)

        listfile2 = open(args.data_cfg, 'r')
        settings_show = listfile2.read().splitlines()
        listfile2.close()

        log = open(log_path, 'w')
        for idx, data in enumerate(settings_show):
            print_log(data, log, display=False)

        tracker_pedestrians = AB3DMOT(cfg=AB3DMOT_cgf,
                                      cat='Pedestrian',
                                      calib=calib,
                                      # Calibration(os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "calib.txt")),
                                      oxts=np.array(poses_seq),  # FrameNumX4X4#imu_poses,
                                      log=log,  # log_file,
                                      ID_init=1,
                                      debug_path=base_save_path)

        tracker_vehicles = AB3DMOT(cfg=AB3DMOT_cgf,
                                   cat='Car',
                                   calib=calib,
                                   # Calibration(os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "calib.txt")),
                                   oxts=np.array(poses_seq),  # FrameNumX4X4#imu_poses,
                                   log=log,  # log_file,
                                   ID_init=1,
                                   debug_path = base_save_path)

        tracker_bikes = AB3DMOT(cfg=AB3DMOT_cgf,
                                cat='Cyclist',
                                calib=calib,
                                # Calibration(os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "calib.txt")),
                                oxts=np.array(poses_seq),  # FrameNumX4X4#imu_poses,
                                log=log,  # log_file,
                                ID_init=1,
                                debug_path = base_save_path)

        # loop over frames
        for idx, point_file in zip(range(len(point_names)), point_names):
            print("{}/{} ".format(idx, len(point_names)), end="", flush=True)
            times = []
            times.append(time.time())
            pose = poses_seq[idx]

            # event :  frames 2938- 3475
            # self.gt_trackID_debug = 22
            # self.gt_classID_debug = "moving-car"
            # self.pred_trackID_debug = [16149,16354,17102,17141,17181,17219,17238,17383,17164,17468,17510,17544,17574,17587,17621,17728,17761]
            # self.pred_trackID_debug = [24]  # we run nirit_tacking at 2980-3202 frames, and it got this new id
            # if idx<2980 or idx >  3202:
            #    continue

            # if idx<2980 or idx >  3475:
            #    continue

            # if idx<2900 or idx >  3475:
            #    continue

            # if idx!=421:
            #    continue

            # if idx!=3319: # filter too small detections
            #    continue

            # if idx !=1638:
            #    continue
            # if idx !=2410:
            #    continue
            # if idx < 3000:
            #   continue
            # if idx > 73:#3959:#3959:#3:#895: jjj
            #    break
            # print("main: frame is ", idx)

            # load current frame
            sem_path = os.path.join(prediction_path, '{0:02d}_{1:07d}.npy'.format(sequence, idx))
            ins_path = os.path.join(prediction_path, '{0:02d}_{1:07d}_i.npy'.format(sequence, idx))
            score_path = os.path.join(prediction_path, '{0:02d}_{1:07d}_c.npy'.format(sequence, idx))

            """
            # upsample paths to files
            upsample_scan_file_path = os.path.join(args.upsample_data,'sequences',str(sequence).zfill(2),'velodyne',str(idx).zfill(6) + '.bin')
            upsample_scan_idx_file_path = os.path.join(args.upsample_data,'sequences',str(sequence).zfill(2),'velodyne',str(idx).zfill(6) + '_idx.npy')
            upsample_labels_path =  os.path.join(args.upsample_data,'sequences',str(sequence).zfill(2),'labels',str(idx).zfill(6) + '.label')


            #upsample data
            upsample_frame_points = np.fromfile(upsample_scan_file_path, dtype=np.float32)
            upsample_points = upsample_frame_points.reshape((-1, 4))

            upsample_scan_idx = np.load(upsample_scan_idx_file_path) # 1 - original point cloud, 0 - addition off SC
            upsample_real_idx = np.where(upsample_scan_idx == 1)[0]

            upsample_label = np.fromfile(upsample_labels_path, dtype=np.uint32)
            upsample_label = upsample_label.reshape((-1))
            upsample_label_sem_class = upsample_label & 0xFFFF  # semantic label in lower half
            #upsample_inst_label = upsample_label >> 16  # instance id in upper half
            """

            label_sem_class = np.load(sem_path)
            label_inst = np.load(ins_path)
            sem_class_scores = np.load(score_path)

            frame_points = np.fromfile(point_file, dtype=np.float32)
            points = frame_points.reshape((-1, 4))
            # hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
            # new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
            # points = new_points[:, :3]

            # filter out close points detection results
            len_x = 3
            len_y = 2
            idx_x = np.where(abs(points[:, 0]) < len_x)[0]
            idx_y = np.where(abs(points[:, 1]) < len_y)[0]
            idx_to_filter = np.intersect1d(idx_x, idx_y)
            """ for debug the filtering
            if idx ==105:
                draw_scenes(points=points,title = 'before filtering close points')
                points = np.delete(points, idx_to_filter, 0)
                draw_scenes(points=points,title = 'after filtering close points')
            """
            label_sem_class[idx_to_filter] = 0  # set those point to the background
            label_inst[idx_to_filter] = 0  # set no label

            # only cars
            # print('frame',idx,' :  np.unique(label_sem_class)  =', np.unique(label_sem_class))
            # things

            #   things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
            #   stuff = [
            #       'road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole',
            #       'traffic-sign'
            #   ]

            # 1.5 scene completion

            if with_sc:
                # s= time.time()
                sc_pred_voxel_coords = np.load(os.path.join(pred_voxel_path, str(idx) + ".voxels.npy"))
                sc_seg_label = np.load(os.path.join(pred_voxel_path, str(idx) + ".sem_label.npy"))
                # print("loadJS3C : time = ", time.time() - s)

            vehicles_str = ["car", "moving-car", "bus", "moving-bus", "truck", "moving-truck", "other-vehicle",
                            "moving-other-vehicle"]
            car_class_id = [id for id, str in det_id2str.items() if str in vehicles_str]
            cars_mask = np.zeros(label_sem_class.shape, dtype=bool)
            for clsID in car_class_id:
                cars_mask = cars_mask | (
                            label_sem_class == clsID)  # if input is from 4DStop, then car class is 1, else car is 10
            if with_sc:
                # car_class_id_global = [id for id, str in config_data['labels'].items() if str in vehicles_str]
                grid_cars_mask = np.zeros(sc_seg_label.shape, dtype=bool)
                for clsID in car_class_id:
                    grid_cars_mask = grid_cars_mask | (sc_seg_label == clsID)

            bikes_str = ["bicycle", "bicyclist", "moving-bicyclist", "motorcycle", "motorcyclist",
                         "moving-motorcyclist"]
            bike_class_id = [id for id, str in det_id2str.items() if str in bikes_str]
            bikes_mask = np.zeros(label_sem_class.shape, dtype=bool)
            for clsID in bike_class_id:
                bikes_mask = bikes_mask | (
                            label_sem_class == clsID)  # if input is from 4DStop, then car class is 2, else car is 10
            if with_sc:
                # bike_class_id_global = [id for id, str in config_data['labels'].items() if str in bikes_str]
                grid_bikes_mask = np.zeros(sc_seg_label.shape, dtype=bool)
                for clsID in bike_class_id:
                    grid_bikes_mask = grid_bikes_mask | (sc_seg_label == clsID)

            Pedestrian_str = ["person", "moving-person"]
            # Pedestrian_str =  ["bicycle", "moving-bicyclist", "motorcycle", "moving-motorcyclist", "bicyclist", "moving-bicyclist","person","moving-person"]
            Pedestrian_class_id = [id for id, str in det_id2str.items() if str in Pedestrian_str]
            Pedestrian_mask = np.zeros(label_sem_class.shape, dtype=bool)
            for clsID in Pedestrian_class_id:
                Pedestrian_mask = Pedestrian_mask | (
                        label_sem_class == clsID)  # if input is from 4DStop, then car class is 2, else car is 10
            if with_sc:
                # Pedestrian_class_id_global = [id for id, str in config_data['labels'].items() if str in Pedestrian_str]
                grid_Pedestrian_mask = np.zeros(sc_seg_label.shape, dtype=bool)
                for clsID in Pedestrian_class_id:
                    grid_Pedestrian_mask = grid_Pedestrian_mask | (sc_seg_label == clsID)

            if with_sc:
                dets_frame_vehicles = get_detections(points, label_inst * cars_mask, label_sem_class, sem_class_scores,
                                                     point_grid=sc_pred_voxel_coords,
                                                     grid_sem=sc_seg_label * grid_cars_mask, minPoints=100)

                dets_frame_bikes = get_detections(points, label_inst * bikes_mask, label_sem_class, sem_class_scores,
                                                  point_grid=sc_pred_voxel_coords,
                                                  grid_sem=sc_seg_label * grid_bikes_mask, minPoints=20)
                dets_frame_pedestrian_before = get_detections(points, label_inst * Pedestrian_mask, label_sem_class,
                                                              sem_class_scores, point_grid=sc_pred_voxel_coords,
                                                              grid_sem=sc_seg_label * grid_Pedestrian_mask,
                                                              minPoints=15)
            else:
                dets_frame_vehicles = get_detections(points, label_inst * cars_mask, label_sem_class, sem_class_scores,
                                                     point_grid=None,
                                                     grid_sem=None, minPoints=None)
                dets_frame_bikes = get_detections(points, label_inst * bikes_mask, label_sem_class, sem_class_scores,
                                                  point_grid=None,
                                                  grid_sem=None, minPoints=None)
                dets_frame_pedestrian_before = get_detections(points, label_inst * Pedestrian_mask, label_sem_class,
                                                              sem_class_scores, point_grid=None,
                                                              grid_sem=None,
                                                              minPoints=None)

            dets_frame_pedestrian = filterPedestrian(dets_frame_pedestrian_before)

            # draw_scenes_raw(points, gt_boxes=dets_frame_pedestrian['dets'], title="scan " + str(
            #    idx) + ' before filtering . blue  = detected')  # raw assumes the format [[h,w,l,x,y,z,theta],...]
            # draw_scenes_raw(points,
            #                gt_boxes=dets_frame_pedestrian['dets'], title="scan " + str(
            #        idx) + '  . blue  = after filtering')  # raw assumes the format [[h,w,l,x,y,z,theta],...]
            # continue

            # for debug purpose: remove chosen detection
            # dets_frame = remove_detections(dets_frame,idx)

            save_detection_file_name = os.path.join(detection_save_path, '%06d.txt' % idx)
            save_detection(save_detection_file_name, dets_frame_vehicles, dets_frame_bikes, dets_frame_pedestrian,
                           det_id2str, idx)

            # YES #dets_frame = get_detections_from_segmentation(points, label_sem_class*cars_mask)
            # NO dets_frame = get_detections_from_segmentation(upsample_points, upsample_label_sem_class * (upsample_label_sem_class ==10)) # cars
            # print('frame', idx, ': detected ', len(dets_frame['dets']),' bounding boxes')

            #		  	dets_all: dict
            #	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
            #	info: a array of other info for each det

            # draw_scenes_raw(points, gt_boxes=dets_frame['dets'],title="scan " + str(idx) + '. blue  = detected') # raw assumes the format [[h,w,l,x,y,z,theta],...]

            # tracking by detection
            since = time.time()
            results_vehicles = tracker_vehicles.track(dets_frame_vehicles, frame=idx,
                                                      seq_name=sequence, points=points[:, :3])
            results_pedestrians = tracker_pedestrians.track(dets_frame_pedestrian, frame=idx,
                                                            seq_name=sequence, points=points[:, :3])
            #results_pedestrians =[[]]
            results_bikes = tracker_bikes.track(dets_frame_bikes, frame=idx,
                                                seq_name=sequence, points=points[:, :3])
            #results_bikes = [[]]
            total_time += time.time() - since

            # save results
            """

            # saving affinity matrix, between the past frame and current frame
            # e.g., for 000006.npy, it means affinity between frame 5 and 6
            # note that the saved value in affinity can be different in reality because it is between the
            # original detections and ego-motion compensated predicted tracklets, rather than between the
            # actual two sets of output tracklets
            save_affi_file = os.path.join(affinity_dir, '%06d.npy' % idx)
            save_affi_vis = os.path.join(affinity_vis, '%06d.txt' % idx)
            if (affi is not None) and (affi.shape[0] + affi.shape[1] > 0):
                # save affinity as long as there are tracklets in at least one frame
                np.save(save_affi_file, affi)

                # cannot save for visualization unless both two frames have tracklets
                if affi.shape[0] > 0 and affi.shape[1] > 0:
                    save_affinity(affi, save_affi_vis)
            """
            # saving trajectories, loop over each hypothesis
            for hypo in range(AB3DMOT_cgf.num_hypo):
                save_trk_file = os.path.join(save_trk_dir[hypo], '%06d.txt' % idx)
                save_trk_file = open(save_trk_file, 'w')

                # detected
                # print('frame {0} detected {1}'.format(idx,str(len(dets_frame['dets'])) ))
                # draw_scenes_raw(points, gt_boxes=dets_frame['dets'],title="main  loop:  scan " + str(idx) + '. blue  = detected')  # raw assumes the format [[h,w,l,x,y,z,theta],...]

                # tracked
                # print('frame {0} tracked {1}'.format(idx, str(len(results[hypo])) ))
                # draw_scenes_raw(points, ref_boxes=results[hypo][:, :7], title="main  loop:  scan " + str(idx) + '. green= tracked')  # raw assumes the format [[h,w,l,x,y,z,theta],...]

                # both
                # draw_scenes_raw(points, gt_boxes=dets_frame['dets'], ref_boxes=results[hypo][:, :7],title = "main  loop: scan " + str(idx) + '. blue  = detected, green= tracked') # raw assumes the format [[h,w,l,x,y,z,theta],...]

                for result_tmp in results_vehicles[hypo]:  # N x 15
                    save_results(result_tmp, save_trk_file, eval_file_dict[hypo],
                                 det_id2str, idx, AB3DMOT_cgf.score_threshold)

                for result_tmp in results_pedestrians[hypo]:  # N x 15
                    save_results(result_tmp, save_trk_file, eval_file_dict[hypo],
                                 det_id2str, idx, AB3DMOT_cgf.score_threshold)

                for result_tmp in results_bikes[hypo]:  # N x 15
                    save_results(result_tmp, save_trk_file, eval_file_dict[hypo],
                                 det_id2str, idx, AB3DMOT_cgf.score_threshold)

                save_trk_file.close()

            """
            ID_start = 1
                        # run tracking for each category
            for cat in cfg.cat_list:
                ID_start = main_per_cat(cfg, cat, log, ID_start)
            """
            print("\r", end='')

        #  end frame loop
        print_log('\nDone!', log=log)
        log.close()

    # end sequence loop


if __name__ == '__main__':
    args = parse_args()
    main(args)
