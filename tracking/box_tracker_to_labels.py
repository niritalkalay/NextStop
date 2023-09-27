import argparse
from collections import defaultdict
import numpy as np
from tracking.tracking_io import parse_boxes
import os
import yaml
from collections import defaultdict
from visual_utils import draw_scenes,draw_scenes_raw ,LaserScanVis # ,Simple_Rendering_Of_Cloud_And_Boxes
from dataclasses import dataclass
from sklearn.neighbors import KDTree
from AB3DMOT_libs.dist_metrics import iou_raw
from sklearn.cluster import AgglomerativeClustering
import open3d as o3d # nirit for debug

input_from_sc= False

@dataclass
class IDmapClass:
    trk_id: list()    # id  in the tracking results
    label_id: list()  # id  in the final results


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

def getpaths(dataset_path, tracking_path, sequence):
    scan_paths = os.path.join(dataset_path, "sequences",
                              sequence, "velodyne")
    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from: \n%s" % scan_paths)
    else:
        print("Sequence folder doesn't exist! Exiting...")
        quit()

    scan_names = []
    for root, dirs, files in os.walk(os.path.expanduser(scan_paths)):
        for file in files:
            if file.lower().endswith('.bin'):
                scan_names.append(os.path.join(root, file))

    scan_names.sort()
    #print('len(scan_names)=', len(scan_names))
    ########################################################

    if os.path.isdir(tracking_path):
        print("Labels folder exists! Using labels from %s" % tracking_path)
    else:
        print("Labels folder doesn't exist! Exiting... ", tracking_path)
        quit()

    track_names = []
    for root, dirs, files in os.walk(os.path.expanduser(tracking_path)):
        for file in files:
            if file.lower().endswith('.txt'):
                track_names.append(os.path.join(root, file))
    track_names.sort()

    assert (len(scan_names)==len(track_names))

    return scan_names, track_names

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum

def findClosest(points,ind1,ind2):

    points1 = points[ind1,:3]
    points2 = points[ind2,:3]

    tree = KDTree(points1, leaf_size=2)
    dist, ind = tree.query(points2, k=int(np.sqrt(len(points1))))

    f_ind = []
    dist_all = []
    ind_all = []
    for ro, d in enumerate(dist):
        for co, dd in enumerate(d):
            dist_all.append(d)
            ind_all.append(ind1[ind[ro, co]])
            if dd < 1:
                f_ind.append(ind1[ind[ro, co]])

    tree = KDTree(points2, leaf_size=2)
    dist, ind = tree.query(points1, k=int(np.sqrt(len(points2))))
    for ro, d in enumerate(dist):
        for co, dd in enumerate(d):
            dist_all.append(d)
            ind_all.append(ind2[ind[ro, co]])
            if dd < 1:
                f_ind.append(ind2[ind[ro, co]])
    f_ind = np.unique(f_ind)
    return f_ind





def getIndToPointsInsideBox_(box,points,label_sem_class,label_inst, point_grid=None, grid_sem=None, showPlots=False):
    h,w,l,cx,cy,cz,theta = box

    x_min = cx - (l) / 2
    y_min = cy - (w) / 2
    z_min = cz - (h) / 2

    x_max = cx + (l) / 2
    y_max = cy + (w) / 2
    z_max = cz + (h) / 2

    idx_x = np.intersect1d(np.where(points[:, 0] <= x_max)[0], np.where(points[:, 0] >= x_min)[0])
    idx_y = np.intersect1d(np.where(points[:, 1] <= y_max)[0], np.where(points[:, 1] >= y_min)[0])
    # since SemanticKitti liDAR located 1.73meter  (- 0.13 error) above the ground , any value larger than this is the ground and need to be filtered.
    idx_z = np.intersect1d(np.where(points[:, 2] <= z_max)[0], np.where(points[:, 2] >= z_min)[0])
    idx = np.intersect1d(np.intersect1d(idx_x, idx_y), idx_z)

    things_idx = np.where((label_sem_class[idx] < 9) & (label_sem_class[idx] > 0))[0]

    res = idx[things_idx]

    if grid_sem is not None and len(res)!=0:
        grid_interesting_idx = np.where(grid_sem != 0)[0]
        if len(grid_interesting_idx) != 0:
            # JS3C range:
            min_extent = [0, -25.6, -2]
            max_extent = [51.2, 25.6, 4.4]
            # to avoid calc kd-tree if tracked object is not in range of sc:
            if ((x_min <= min_extent[0] <= x_max) or (x_min <= max_extent[0] <= x_max) or (
                    x_min >= min_extent[0] and x_max <= max_extent[0])) and \
                    ((y_min <= min_extent[1] <= y_max) or (y_min <= max_extent[1] <= y_max) or (
                            y_min >= min_extent[1] and y_max <= max_extent[1])):

                obj_points  = points[res,:3]

                #draw_scenes(points=obj_points, title='obj_points in the range of sc')
                #draw_scenes(points=point_grid[grid_interesting_idx], title='point_grid[grid_interesting_idx]')

                # find closest point_grid to the object
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
                # np.min(dist_all)
                # np.max(dist_all)
                f_ind = np.unique(f_ind)

                #draw_scenes(points=point_grid[grid_interesting_idx], title='point_grid[grid_interesting_idx]')

                if len(f_ind) >= 100:#minPoints:

                    obj_points_with_sc = np.vstack((obj_points, point_grid[grid_interesting_idx[f_ind]]))
                    #draw_scenes(points=obj_points_with_sc, title='obj_points with no sc')

                    new_x_min,new_y_min,new_z_min =  np.min(obj_points_with_sc,axis=0)
                    new_x_max, new_y_max, new_z_max = np.max(obj_points_with_sc, axis=0)

                    idx_x = np.intersect1d(np.where(points[:, 0] <= new_x_max)[0], np.where(points[:, 0] >= new_x_min)[0])
                    idx_y = np.intersect1d(np.where(points[:, 1] <= new_y_max)[0], np.where(points[:, 1] >= new_y_min)[0])
                    idx_z = np.intersect1d(np.where(points[:, 2] <= new_z_max)[0], np.where(points[:, 2] >= new_z_min)[0])
                    idx = np.intersect1d(np.intersect1d(idx_x, idx_y), idx_z)

                    show_plot  = True
                    things_idx = np.where((label_sem_class[idx] < 9) & (label_sem_class[idx] > 0))[0]
                    if len(np.unique(label_inst[res])) != len(np.unique(label_inst[idx[things_idx]])):
                        draw_scenes(points=point_grid[grid_interesting_idx], title='point_grid[grid_interesting_idx]')
                        print ("larger!")
                        show_plot = True


                    if show_plot:
                        draw_scenes(points=points[res, :3], title='points no sc')
                    res = idx[things_idx]
                    if show_plot:
                        draw_scenes(points=points[res,:3], title='points with sc')
                        #print("")


    if len(res)!=0:
        # find the closest label to center box
        # (using the closest label to handle the box size mismatch)

        tree = KDTree(points[res, :3], leaf_size=2)
        dist, ind = tree.query(np.array([cx, cy, cz]).reshape(1, -1), k=1)
        closest_label = label_inst[res[ind]][0]
        res = np.union1d(np.where(label_inst == closest_label)[0], res)
        """
        unique_labels_inside_box = np.unique(label_inst[idx[things_idx]])
        # insert second closest label
        if len(unique_labels_inside_box)>1:

            left_labels_points = np.where(  label_inst[idx[things_idx]]!=closest_label)[0]
            tree = KDTree(points[left_labels_points,:3], leaf_size=2)
            dist, ind = tree.query(np.array([cx, cy, cz]).reshape(1, -1), k=1)
            
            if dist < 1.5:

                second_close_label = label_inst[left_labels_points[ind]][0]
                all_second_ind = np.where(label_inst == second_close_label)[0]
                inside_box_second_ind = np.where(label_inst[left_labels_points] == second_close_label)[0]
                if len(inside_box_second_ind) > 20: #/  len(all_second_ind) > 0.3:
                    res = np.union1d(all_second_ind,res)
                    print("here!")

                    if True:#showPlots:
                        draw_scenes_raw(points=points, ref_boxes=[box], title='input : ')
                        draw_scenes_raw(points=points[idx], ref_boxes=[box],
                                        title='points[idx] : ')
                        draw_scenes_raw(points=points[idx[things_idx]], ref_boxes=[box],
                                        title='things inside box : ')

                        draw_scenes_raw(points=points[np.where(label_inst == closest_label)[0]], ref_boxes=[box],
                                        title='closest instances : ')
                        draw_scenes_raw(points=points[np.where(label_inst == second_close_label)[0]], ref_boxes=[box],
                                        title='second close instances : ')

                        # draw_scenes_raw(points=points[idx[to_add]], ref_boxes=[box],
                        #                title='to_add inside box : ')
                        draw_scenes_raw(points=points[res], ref_boxes=[box], title='out : ')
        """


    else:
        return res

    if showPlots:
        draw_scenes_raw(points=points, ref_boxes=[box], title='input : ')
        draw_scenes_raw(points=points[idx], ref_boxes=[box],
                        title='points[idx] : ')
        draw_scenes_raw(points=points[idx[things_idx]], ref_boxes=[box],
                        title='things inside box : ')

        draw_scenes_raw(points=points[np.where(label_inst == closest_label)[0]], ref_boxes=[box],
                        title='closest instances : ')

        #draw_scenes_raw(points=points[idx[to_add]], ref_boxes=[box],
        #                title='to_add inside box : ')
        draw_scenes_raw(points=points[res], ref_boxes=[box], title='out : ')
        #draw_scenes_raw(points=points[res2], ref_boxes=[box], title='out2 : ')

    return res

def getIndToPointsInsideBox_gt(box,points,label_sem_class,label_inst,showPlots=False):
    h,w,l,cx,cy,cz,theta = box

    x_min = cx - (l) / 2
    y_min = cy - (w) / 2
    z_min = cz - (h) / 2

    x_max = cx + (l) / 2
    y_max = cy + (w) / 2
    z_max = cz + (h) / 2

    idx_x = np.intersect1d(np.where(points[:, 0] <= x_max)[0], np.where(points[:, 0] >= x_min)[0])
    idx_y = np.intersect1d(np.where(points[:, 1] <= y_max)[0], np.where(points[:, 1] >= y_min)[0])
    # since SemanticKitti liDAR located 1.73meter  (- 0.13 error) above the ground , any value larger than this is the ground and need to be filtered.
    idx_z = np.intersect1d(np.where(points[:, 2] <= z_max)[0], np.where(points[:, 2] >= z_min)[0])
    idx = np.intersect1d(np.intersect1d(idx_x, idx_y), idx_z)

    things_idx = np.where((label_sem_class[idx] < 9) & (label_sem_class[idx] > 0))[0]
    res =idx[things_idx]

    return res

def getIndToPointsInsideBox_simple(box,points,showPlots=False):
    h,w,l,cx,cy,cz,theta = box

    x_min = cx - (l) / 2
    y_min = cy - (w) / 2
    z_min = cz - (h) / 2

    x_max = cx + (l) / 2
    y_max = cy + (w) / 2
    z_max = cz + (h) / 2

    idx_x = np.intersect1d(np.where(points[:, 0] <= x_max)[0], np.where(points[:, 0] >= x_min)[0])
    idx_y = np.intersect1d(np.where(points[:, 1] <= y_max)[0], np.where(points[:, 1] >= y_min)[0])
    idx_z = np.intersect1d(np.where(points[:, 2] <= z_max)[0], np.where(points[:, 2] >= z_min)[0])
    idx = np.intersect1d(np.intersect1d(idx_x, idx_y), idx_z)

    if showPlots:
        draw_scenes_raw(points=points, ref_boxes=[box], title='input : ')
        draw_scenes_raw(points=points[idx], ref_boxes=[box], title='out : ')
    return idx

def getIndToPointsInsideBox(box,points,label_sem_class,showPlots=False):
    h,w,l,cx,cy,cz,theta = box

    x_min = cx - (l+0.5) / 2
    y_min = cy - (w+0.5) / 2
    z_min = cz - (h+0.5) / 2

    x_max = cx + (l+0.5) / 2
    y_max = cy + (w+0.5) / 2
    z_max = cz + (h+0.5) / 2
    """
    
    x_min = cx - (l) / 2
    y_min = cy - (w) / 2
    z_min = cz - (h) / 2

    x_max = cx + (l) / 2
    y_max = cy + (w) / 2
    z_max = cz + (h) / 2
    """
    idx_x = np.intersect1d(np.where(points[:, 0] <= x_max)[0], np.where(points[:, 0] >= x_min)[0])
    idx_y = np.intersect1d(np.where(points[:, 1] <= y_max)[0], np.where(points[:, 1] >= y_min)[0])
    # since SemanticKitti liDAR located 1.73meter  (- 0.13 error) above the ground , any value larger than this is the ground and need to be filtered.
    idx_z = np.intersect1d(np.where(points[:, 2] <= z_max)[0], np.where(points[:, 2] >= z_min)[0])
    idx = np.intersect1d(np.intersect1d(idx_x, idx_y), idx_z)
    things_idx = np.where((label_sem_class[idx] < 9) & (label_sem_class[idx] > 0))[0]

    if showPlots:
        draw_scenes_raw(points=points, ref_boxes=[box], title='input : ')
        draw_scenes_raw(points=points[idx[things_idx],:], ref_boxes=[box], title='out : ')
    return idx[things_idx]

def getIndToPointsInsideBoxWithUncertainty(box,points,label_sem_class,offset=0.0,showPlots=False):
    h,w,l,cx,cy,cz,theta = box


    # middle box - the tracker box
    x_min_tracker = cx - (l) / 2
    y_min_tracker = cy - (w) / 2
    z_min_tracker = cz - (h) / 2

    x_max_tracker = cx + (l) / 2
    y_max_tracker = cy + (w) / 2
    z_max_tracker = cz + (h) / 2

    # exterior box
    x_min_exterior = x_min_tracker - offset * l
    y_min_exterior = y_min_tracker - offset * w
    z_min_exterior = z_min_tracker - offset * h

    x_max_exterior = x_max_tracker + offset * l
    y_max_exterior = y_max_tracker + offset * w
    z_max_exterior = z_max_tracker + offset * h

    # interior box
    x_min_interior = x_min_tracker #+ offset * l
    y_min_interior = y_min_tracker #+ offset * w
    z_min_interior = z_min_tracker #+ offset * h

    x_max_interior = x_max_tracker #- offset * l
    y_max_interior = y_max_tracker #- offset * w
    z_max_interior = z_max_tracker #- offset * h

    # points_ind_inside_interior
    idx_x = np.intersect1d(np.where(points[:, 0] <= x_max_interior)[0],
                           np.where(points[:, 0] >= x_min_interior)[0])
    idx_y = np.intersect1d(np.where(points[:, 1] <= y_max_interior)[0],
                           np.where(points[:, 1] >= y_min_interior)[0])
    idx_z = np.intersect1d(np.where(points[:, 2] <= z_max_interior)[0],
                           np.where(points[:, 2] >= z_min_interior)[0])
    idx = np.intersect1d(np.intersect1d(idx_x, idx_y), idx_z)
    things_idx = np.where((label_sem_class[idx] < 9) & (label_sem_class[idx] > 0))[0]
    points_ind_inside_interior=idx[things_idx]

    # points_ind_inside_exterior
    idx_x = np.intersect1d(np.where(points[:, 0] <= x_max_exterior)[0],
                           np.where(points[:, 0] >= x_min_exterior)[0])
    idx_y = np.intersect1d(np.where(points[:, 1] <= y_max_exterior)[0],
                           np.where(points[:, 1] >= y_min_exterior)[0])
    idx_z = np.intersect1d(np.where(points[:, 2] <= z_max_exterior)[0],
                           np.where(points[:, 2] >= z_min_exterior)[0])
    idx = np.intersect1d(np.intersect1d(idx_x, idx_y), idx_z)
    things_idx = np.where((label_sem_class[idx] < 9) & (label_sem_class[idx] > 0))[0]
    points_ind_inside_exterior=idx[things_idx]

    # points_ind_inside_tracker
    idx_x = np.intersect1d(np.where(points[:, 0] <= x_max_tracker)[0],
                           np.where(points[:, 0] >= x_min_tracker)[0])
    idx_y = np.intersect1d(np.where(points[:, 1] <= y_max_tracker)[0],
                           np.where(points[:, 1] >= y_min_tracker)[0])
    idx_z = np.intersect1d(np.where(points[:, 2] <= z_max_tracker)[0],
                           np.where(points[:, 2] >= z_min_tracker)[0])
    idx = np.intersect1d(np.intersect1d(idx_x, idx_y), idx_z)
    things_idx = np.where((label_sem_class[idx] < 9) & (label_sem_class[idx] > 0))[0]
    points_ind_inside_tracker = idx[things_idx]

    uncertainty_points_ind = np.setxor1d(points_ind_inside_exterior, points_ind_inside_interior, assume_unique=False)

    if showPlots:
        draw_scenes_raw(points=points, ref_boxes=[box], title='input : ')
        draw_scenes_raw(points=points[points_ind_inside_interior,:], ref_boxes=[box], title='inner points : ')
        draw_scenes_raw(points=points[points_ind_inside_tracker, :], ref_boxes=[box], title='tracker points : ')
        draw_scenes_raw(points=points[points_ind_inside_exterior, :], ref_boxes=[box], title='exterior points : ')
        draw_scenes_raw(points=points[uncertainty_points_ind, :], ref_boxes=[box], title='uncertainty points : ')

    return points_ind_inside_interior , uncertainty_points_ind


def detectOverlappingBoxes(array_of_boxes):
    nbox = len(array_of_boxes)
    overlapping_boxes = np.array([])
    for i in range(nbox): # range(start, stop, step)
        box_a = array_of_boxes[i]
        for j in range(i+1,nbox):
            box_b = array_of_boxes[j]
            """
            draw_scenes_raw(points= np.zeros((1,4)), ref_boxes=[box_a],
                            gt_boxes=[box_b], title="detectOverlappingBoxes, box_a green, box_b blue, box_a,box_b =" + str(i) +"," + str(j))
            """
            d = iou_raw(box_a, box_b, metric='iou_3d')
            #print("i,j,d = ",i," , ",j," , ",d)
            if d > 0.001:
                overlapping_boxes = np.vstack((overlapping_boxes,np.expand_dims(np.array((i,j)),axis=0))) if len(
                    overlapping_boxes) != 0 else np.expand_dims(np.array((i,j)),axis=0)

    non_overlapping_boxes =[]
    overlap_as_list = np.unique(overlapping_boxes.ravel())
    for i in range(nbox):
        if i not in overlap_as_list:
            non_overlapping_boxes.append(i)

    return overlapping_boxes, non_overlapping_boxes

def detectOverlappingPoints(points,array_of_boxes,array_of_points_of_boxes):
    nbox = len(array_of_points_of_boxes)
    overlapping_boxes = np.array([])
    for i in range(nbox): # range(start, stop, step)
        box_a_points_ind = array_of_points_of_boxes[i]
        box_a = array_of_boxes[i]
        for j in range(i+1,nbox):
            box_b_points_ind = array_of_points_of_boxes[j]
            box_b = array_of_boxes[j]
            overlap_ind = np.intersect1d(box_a_points_ind,box_b_points_ind)
            if len(overlap_ind)>3:
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

                overlapping_boxes = np.vstack((overlapping_boxes,np.expand_dims(np.array((i,j)),axis=0))) if len(
                    overlapping_boxes)>3 else np.expand_dims(np.array((i,j)),axis=0)

    non_overlapping_boxes =[]
    overlap_as_list = np.unique(overlapping_boxes.ravel())
    for i in range(nbox):
        if i not in overlap_as_list:
            non_overlapping_boxes.append(i)

    return overlapping_boxes, non_overlapping_boxes

def getIndToPointsInsideBoxes (array_of_boxes,points,
                               label_sem_class,label_inst,
                               array_of_ClassID,
                               sc_pred_voxel_coords=None,
                               sc_seg_label=None):

    results = [None] * len(array_of_boxes)

    show_plot=False
    for b,box in enumerate(array_of_boxes):
        #if b in [9]:
        #    show_plot=True
        #else:
        #    show_plot=False

        results[b] = getIndToPointsInsideBox_(box, points,
                                              label_sem_class, label_inst,
                                              sc_pred_voxel_coords,sc_seg_label,
                                              show_plot)

    overlapping_boxes, non_overlapping_boxes = detectOverlappingPoints(points, array_of_boxes,
                                                                       results)  # there is overlap if points overlap
    # refine overlapping boxes
    for i in range(len(overlapping_boxes)):
        overlapping_box0 = overlapping_boxes[i, 0]
        overlapping_box1 = overlapping_boxes[i, 1]

        points_only_in_box0 = getIndToPointsInsideBox_simple(box=array_of_boxes[overlapping_box0],
                                                             points=points)
        points_only_in_box1 = getIndToPointsInsideBox_simple(box=array_of_boxes[overlapping_box1],
                                                             points=points)

        common = np.intersect1d(points_only_in_box0, points_only_in_box1)
        common_of_box0 = len(common) / len(points_only_in_box0)
        common_of_box1 = len(common) / len(points_only_in_box1)

        #print("common_of_box0: ", common_of_box0)
        #print("common_of_box1: ", common_of_box1)

        if common_of_box1 > common_of_box0:
            common = np.intersect1d(results[overlapping_box0], results[overlapping_box1])
            # take all common points to box 1,this is the same as  remove common points from box 0
            results[overlapping_box0] = np.array(
                [el for el in results[overlapping_box0] if el not in common])
        else:  # common_of_box0 > 0.8:
            # take all common points to box 0,this is the same as  remove common points from box 1
            common = np.intersect1d(results[overlapping_box0], results[overlapping_box1])
            results[overlapping_box1] = np.array(
                [el for el in results[overlapping_box1] if el not in common])

    return results


def getIndToPointsInsideBoxes_(array_of_boxes, points, label_sem_class, label_inst,sc_pred_voxel_coords=None, sc_seg_label=None):
    results = [None] * len(array_of_boxes)

    show_plot = False
    for b, box in enumerate(array_of_boxes):
        #if b==7 :#or b ==6:
        #    show_plot=True
        #else:
        #    show_plot=False
        results[b] = getIndToPointsInsideBox_(box, points, label_sem_class, label_inst, show_plot)

    overlapping_boxes, _ = detectOverlappingPoints(points, array_of_boxes,
                                                   results)  # there is overlap if points overlap
    # overlapping_boxes, _ = detectOverlappingBoxes(array_of_boxes) # there is overlap if boxes overlap

    for i in range(len(overlapping_boxes)):
        overlapping_box0 = overlapping_boxes[i, 0]
        overlapping_box1 = overlapping_boxes[i, 1]
        common = np.intersect1d(results[overlapping_box0], results[overlapping_box1])
        if len(common) > 2:
            # https://scikit-learn.org/stable/modules/clustering.html#clustering# https://scikit-learn.org/stable/modules/clustering.html#clustering
            union = np.union1d(results[overlapping_box0], results[overlapping_box1])

            h0, w0, l0, cx0, cy0, cz0, theta0 = array_of_boxes[overlapping_box0]
            h1, w1, l1, cx1, cy1, cz1, theta1 = array_of_boxes[overlapping_box1]

            if (h0 * w0 * l0) > (h1 * w1 * l1):
                # box0 is bigger than box1
                # fill box0 first
                # then fill box1
                # meaning overlap area belongs to box 1
                # so remove overlap points from box0

                # draw_scenes_raw(points[union], ref_boxes=[array_of_boxes[overlapping_box0]],
                #                gt_boxes=[array_of_boxes[overlapping_box1]], title="all points on both boxes")
                # draw_scenes_raw(points[results[overlapping_box0]], ref_boxes=[array_of_boxes[overlapping_box0]],
                #                gt_boxes=[array_of_boxes[overlapping_box1]], title="points of box0")
                # draw_scenes_raw(points[results[overlapping_box1]], ref_boxes=[array_of_boxes[overlapping_box0]],
                #                gt_boxes=[array_of_boxes[overlapping_box1]], title="points of box1")

                # draw_scenes_raw(points[results[overlapping_box0]], ref_boxes=[array_of_boxes[overlapping_box0]], title="box0 before")
                # draw_scenes_raw(points[common], ref_boxes=[array_of_boxes[overlapping_box0]],
                #                title="common on box 0 ")

                results[overlapping_box0] = np.array([el for el in results[overlapping_box0] if
                                                      el not in common])
                # draw_scenes_raw(points[results[overlapping_box0]], ref_boxes=[array_of_boxes[overlapping_box0]],
                #                title="box0 afterr")
            else:
                # box1 is bigger than box00
                # fill box1 first
                # then fill box0
                # meaning overlap area belongs to box 0
                # so remove overlap points from box1
                results[overlapping_box1] = np.array([el for el in results[overlapping_box1] if el not in common])

            #draw_scenes_raw(points[union], ref_boxes=[array_of_boxes[overlapping_box0]],
            #                gt_boxes=[array_of_boxes[overlapping_box1]], title="all points on both boxes")
            #draw_scenes_raw(points[results[overlapping_box0]], ref_boxes=[array_of_boxes[overlapping_box0]],
            #                gt_boxes=[array_of_boxes[overlapping_box1]], title="points of box0")
            #draw_scenes_raw(points[results[overlapping_box1]], ref_boxes=[array_of_boxes[overlapping_box0]],
            #                gt_boxes=[array_of_boxes[overlapping_box1]], title="points of box1")

            # draw_scenes_raw(points[common], ref_boxes=[array_of_boxes[overlapping_box0]],gt_boxes=[array_of_boxes[overlapping_box1]], title="common points on both boxes")
            # draw_scenes_raw(points[union], ref_boxes=[array_of_boxes[overlapping_box0]],
            #                gt_boxes=[array_of_boxes[overlapping_box1]], title="union points on both boxes")

            """

            #cluster_ = cluster.Birch(n_clusters=2, threshold = .65).fit(points[common]) #points[common])
            cluster_ = cluster.AgglomerativeClustering(n_clusters=2).fit(points[common]) #points[common])
            #cluster_ = cluster.KMeans(n_clusters=2, random_state=100).fit(points[union,:2])
 
            if len(np.unique(cluster_.labels_)) <2:
                print("Birch did not worked")
                cluster_ = cluster.KMeans(n_clusters=2, random_state=50).fit(points[common])
 
            cluster_1_ind =  union[np.where(cluster_.labels_ == 0)[0]]
            cluster_2_ind =  union[np.where(cluster_.labels_ == 1)[0]]
            if True:#overlapping_box0==4 or overlapping_box1 ==4:
              draw_scenes_raw(points[union],ref_boxes=[array_of_boxes[overlapping_box0]],gt_boxes=[array_of_boxes[overlapping_box1]],title="all points on both boxes")
              draw_scenes_raw(points[cluster_1_ind], ref_boxes=[array_of_boxes[overlapping_box0]],
                              gt_boxes=[array_of_boxes[overlapping_box1]], title="common points on cluster 1")
              draw_scenes_raw(points[cluster_2_ind], ref_boxes=[array_of_boxes[overlapping_box0]],
                              gt_boxes=[array_of_boxes[overlapping_box1]], title="common points on cluster 2")
 
 
            # PointsSimilarity calculates the mean distance between two set of point.
            # min value -> the closer those groups are
            dist_box0_to_cluster1= PointsSimilarity(points[results[overlapping_box0]], points[cluster_1_ind])
            dist_box0_to_cluster2 = PointsSimilarity(points[results[overlapping_box0]], points[cluster_2_ind])
 
            dist_box1_to_cluster1 = PointsSimilarity(points[results[overlapping_box1]], points[cluster_1_ind])
            dist_box1_to_cluster2 = PointsSimilarity(points[results[overlapping_box1]], points[cluster_2_ind])
 
            tmp_array = np.array ([dist_box0_to_cluster1,dist_box0_to_cluster2 ,dist_box1_to_cluster1, dist_box1_to_cluster2 ])
            min_ind = np.argmin(tmp_array)
 
            if min_ind == 0 or min_ind == 3:
                # box0 -> cluster2
                # box1 -> cluster1
                results[overlapping_box0] = np.array([el for el in results[overlapping_box0] if
                                                       el not in cluster_1_ind])
                results[overlapping_box1] = np.array([el for el in results[overlapping_box1] if
                                                       el not in cluster_2_ind])
 
            elif min_ind == 1 or min_ind == 2:
                # box0 -> cluster1
                # box1 -> cluster2
                results[overlapping_box0] = np.array([el for el in results[overlapping_box0] if
                                                      el not in cluster_2_ind])
                results[overlapping_box1] = np.array([el for el in results[overlapping_box1] if
                                                      el not in cluster_1_ind])
            else:
                 print(" error should not happen!")
 
            """
            """
            # option A : kbtree
            tree_box0 = KDTree(points[results[overlapping_box0],:3], leaf_size=2)
            tree_box1 = KDTree(points[results[overlapping_box1], :3], leaf_size=2)
 
            box0_to_culster1_dist ,nearest_ind_box0_to_culster1 = tree_box0.query(points[cluster_1_ind,:3] , k = np.min((10,len(cluster_1_ind))))
            box0_to_culster2_dist, nearest_ind_box0_to_culster2 = tree_box0.query(points[cluster_2_ind, :3],k=np.min((10, len(cluster_2_ind))))
 
            box1_to_culster1_dist, nearest_ind_box1_to_culster1 = tree_box1.query(points[cluster_1_ind, :3],
                                                                                  k=np.min((10, len(cluster_1_ind))))
            box1_to_culster2_dist, nearest_ind_box1_to_culster2 = tree_box1.query(points[cluster_2_ind, :3],
                                                                                  k=np.min((10, len(cluster_2_ind))))
 
            sum_box0_to_cluster1 = np.sum(np.sum(box0_to_culster1_dist))
            sum_box1_to_cluster1 = np.sum(np.sum(box1_to_culster1_dist))
 
            sum_box0_to_cluster2 = np.sum(np.sum(box0_to_culster2_dist))
            sum_box1_to_cluster2 = np.sum(np.sum(box1_to_culster2_dist))
 
            # who get cluster 1
            if sum_box0_to_cluster1 < sum_box1_to_cluster1 :
                # box 0 has closer distance to cluster1
                results[overlapping_box0] = np.array([el for el in results[overlapping_box0] if
                                             el not in cluster_2_ind])
            else:
               # box 1 has closer distance to cluster1
               results[overlapping_box1] = np.array([el for el in results[overlapping_box1] if
                                                     el not in cluster_2_ind])
 
            # who got cluster 2
            if sum_box0_to_cluster2 < sum_box1_to_cluster2 :
                # box 0 has closer distance to cluster2
                results[overlapping_box0] = np.array([el for el in results[overlapping_box0] if
                                             el not in cluster_1_ind])
            else:
               # box 1 has closer distance to cluster2
               results[overlapping_box1] = np.array([el for el in results[overlapping_box1] if
                                                     el not in cluster_1_ind])
            """
            # if True:#overlapping_box0 == 4 or overlapping_box1 == 4:
            # draw_scenes_raw(points[results[overlapping_box0].astype(np.int)],
            #                 ref_boxes=[array_of_boxes[overlapping_box0]], title="box0, after clustering")
            # draw_scenes_raw(points[results[overlapping_box1].astype(np.int)],
            #                 ref_boxes=[array_of_boxes[overlapping_box1]],title="box1 after clustering")

    return results

def init(sequence,split,path_to_gt,seg_prediction_dir,box_tracker_path):
    # 1. get file names & path
    if split == 'valid':
        prediction_path = '{}/val_probs'.format(seg_prediction_dir)
    else:
        prediction_path = '{}/probs'.format(seg_prediction_dir)


    #path_to_track_result = '/media/nirit/mugiwara/code/4D-StOP/media/nirit/mugiwara/code/4D-StOP/test/Log_2022-06-13_17-33-24_importance_None_str1_bigpug_2_current_chkp/AB3DMOT_tracker/predictions/trk_withid_0/sequences08'
    # new path

    path_to_track_result = os.path.join(box_tracker_path,"predictions","trk_withid_0","sequences{0:02d}".format(sequence))
    save_path = os.path.join(box_tracker_path,'to_labels')
    #pth_js3c_output = '/media/nirit/mugiwara/code/JS3C-Net/JS3C-Net-main/nirit_upsample_cloud/raw_results'

    #sc_path = os.path.join(pth_js3c_output, "sequences", '{0:02d}'.format(sequence))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    next_path = os.path.join(save_path,'sequences')
    if not os.path.exists(next_path):
        os.mkdir(next_path)
    next_path = os.path.join(next_path,'{:02d}'.format(sequence))
    if not os.path.exists(next_path):
        os.mkdir(next_path)
    next_path = os.path.join(next_path, 'predictions')
    if not os.path.exists(next_path):
        os.mkdir(next_path)

    scan_names, track_names  = getpaths(path_to_gt, path_to_track_result, sequence='{0:02d}'.format(int(sequence)))


    # config data
    data_cfg =os.path.join(path_to_gt,"semantic-kitti.yaml")
    cfg= yaml.safe_load(open(data_cfg, 'r'))

    return save_path, scan_names, prediction_path, track_names, cfg


def main(args):
    args = parse_args()

    save_path, scan_names, prediction_path, track_names, config_data = init(args.sequences,
                                                                            args.split,
                                                                            args.dataset,
                                                                            args.predictions,
                                                                            os.path.join(args.predictions,'NextStop_tracker'))

    classStr2Int_global = {value: key for key, value in config_data['labels'].items()} # global labels
    classInt2Str_global = config_data['labels'] # global labels
    classStr2Int_local = {config_data['labels'][value] : key for key, value in config_data['learning_map_inv'].items()}  #
    classInt2Str_local = {key: config_data['labels'][value] for key, value in config_data['learning_map_inv'].items()} # lables match 4dStop net learning, called it in local map

    learning_map_doc = config_data['learning_map']
    inv_learning_map = np.zeros((np.max([k for k in config_data['learning_map_inv'].keys()]) + 1), dtype=np.int32)
    for k, v in config_data['learning_map_inv'].items():
        inv_learning_map[k] = v

    # 2.2 init helpers structures
    IDMemory = [IDmapClass(trk_id=[], label_id=[]) for _ in range(3)]
    vehicles_str = ["car", "moving-car", "bus", "moving-bus", "truck", "moving-truck", "other-vehicle",
                    "moving-other-vehicle"]
    bikes_str = ["bicycle", "bicyclist", "moving-bicyclist", "motorcycle", "motorcyclist", "moving-motorcyclist"]
    Pedestrian_str = ["person", "moving-person"]



    """
    # fix segmentation based on tracking
    #trackMem = [trackmapClass(trk_id=[], label_id=[]) for _ in range(3)]
    from collections import defaultdict
    trackMem =[defaultdict(list) for _ in range(3)]

    #class trackInfo:
    #    id: int
    #    cls: list()
    #    data: list()
    ##results = {value: [] for key, value in config_data['labels'].items()}
    # results[class_str].append((info(id=array_of_TrackedID[i], frames=[frameNum],box_volume =[vol])))
    print("loading track results to fix segmentation based on majority")
    for track_file in track_names:
        array_of_boxes, array_of_TrackedID, array_of_ClassID, array_of_DetectionScore = parse_boxes(filename=track_file)
        for b in range(len(array_of_boxes)):
           track_id = array_of_TrackedID[b]
           cls_str = array_of_ClassID[b]

           if cls_str in vehicles_str:
               cls_group_ind_in_trackmem = 0
           elif cls_str in bikes_str:
               cls_group_ind_in_trackmem = 1
           elif cls_str in Pedestrian_str:
               cls_group_ind_in_trackmem = 2
           else :
               raise ValueError('unsupported class', cls_str)

           trackMem[cls_group_ind_in_trackmem][track_id].append(cls_str)
    print(".. DONE! ")
    """
    new_id = 1 + 300 # + 300, so that when doing instance label we won't have the same id as to the stuff ID.

    print("START :")
    for idx in range(len(scan_names)):
        print("{}/{} ".format(idx, len(scan_names)), end="", flush=True)

        #if idx!=136: # car next to me - problem with height
        #    continue
        #if idx!=93:  # bike next to me - problem with road
        #    continue

        #if idx!=2772:#1246: # car next to me - problem with height
        #  continue

        #if idx !=7: #2772:#3006:  # to debug  filling close objects
        #  continue

        #if idx not in [2092,2093]:#1230:#580: #"#3973:#2772: #3972#2772:#3006:  # to debug  two tracker overlap (cars gt 395,237)
        # continue

        # gt car: 237 frames 3920-4070 and gt car 395 frames 3922-4070
        # interesting to debug two tracker overlap
        #if idx not in [2709]:#[2684]:#[2612]:#[2573]:#[2544]:#[2163]:#[2127]:#[1244]:#[559]:#[3965]:#[3973]:#[3973]: #[3964]: #[3965]:
        #  continue
        #if idx != 3972:#"#!= 13:#3965:#3976:#149:#3965:#2612:#3999:#3973:#3999:#3965:
        #  continue
        #print("idx =", idx)

        # 1.  load inputs:
        # 1.1 points
        point_file = scan_names[idx]
        frame_points = np.fromfile(point_file, dtype=np.float32)
        points = frame_points.reshape((-1, 4))
        # 1.2 segm
        sem_path = os.path.join(prediction_path, '{0:02d}_{1:07d}.npy'.format(args.sequences, idx))
        label_sem_class = np.load(sem_path) # in 4d-Stop numbering
        # 1.3 instance
        ins_path = os.path.join(prediction_path, '{0:02d}_{1:07d}_i.npy'.format(args.sequences, idx))
        label_inst = np.load(ins_path)
        # 1.4 tracking results
        track_file = track_names[idx]
        array_of_boxes, array_of_TrackedID, array_of_ClassID,array_of_DetectionScore = parse_boxes(filename=track_file)
        # np.array((float(h), float(w), float(l), float(x), float(y), float(z),float(theta)))

        # 2. init results
        new_sem_label  = np.ones (len(points),dtype=np.int32) * -1
        new_inst_label = np.ones (len(points),dtype=np.int32) * -1

        # 3. fill points inside the bounding box
        list_of_ind_points_inside_box = getIndToPointsInsideBoxes(array_of_boxes,
                                                                  points,
                                                                  label_sem_class,label_inst,
                                                                  array_of_ClassID,
                                                                  sc_pred_voxel_coords=None,
                                                                  sc_seg_label=None)


        for b in range(len(array_of_boxes)):
           cls_str = array_of_ClassID[b]

           if cls_str in vehicles_str:
               cls_group_ind_in_mem = 0
           elif cls_str in bikes_str:
               cls_group_ind_in_mem = 1
           elif cls_str in Pedestrian_str:
               cls_group_ind_in_mem = 2
           else :
               raise ValueError('unsupported class', cls_str)

           track_id = array_of_TrackedID[b]
           #if track_id in trackMem[cls_group_ind_in_trackmem]:
           #    cls_str,count_appearances = find_majority(trackMem[cls_group_ind_in_mem][track_id])
               #if count_appearances < 7:
               #    print("ignoring trackID =",track_id, " of class ",cls_str ,"since  count_appearances = ",count_appearances )
               #    continue



           cls_id_global_map =  classStr2Int_global[cls_str]
           #cls_id_local_map =  classStr2Int_local[cls_str]

           #box = array_of_boxes[b]

           unmarked_points_idx = np.where(new_inst_label==-1)[0]
           if len(unmarked_points_idx!=0):
               object_index =  list_of_ind_points_inside_box[b] #getPointsInsideBox(box,points[unmarked_points_idx])
               if len(object_index)!=0:
                   #things_idx = np.where((label_sem_class[object_index] < 9) & (label_sem_class[object_index] > 0))[0]
                   idx_to_fill = object_index# [things_idx.astype(int)]#unmarked_points_idx[object_index]
                   # check if this is the first time we encountered this TrackID for this class
                   idx_exist_in_mem = np.where(IDMemory[cls_group_ind_in_mem].trk_id==track_id)[0]
                   if len(idx_exist_in_mem) == 0:
                       # first appearance of this track_id for this class
                       new_inst_label[idx_to_fill] = new_id
                       new_sem_label [idx_to_fill] = cls_id_global_map
                       # add to memory
                       IDMemory[cls_group_ind_in_mem].trk_id.append(track_id)
                       IDMemory[cls_group_ind_in_mem].label_id.append(new_id)
                       # increase the new inst label for next filling
                       new_id += 1
                   else:
                       new_inst_label[idx_to_fill] = IDMemory[cls_group_ind_in_mem].label_id[int(idx_exist_in_mem)]
                       new_sem_label[idx_to_fill] = cls_id_global_map
           else:
               print("no more points to fill!")
               break
        # END BOX LOOP


        # fill the missing seg results
        idx_to_fill= np.where(new_sem_label==-1)[0]
        if len(idx_to_fill)!=0:
            new_sem_label[idx_to_fill]= inv_learning_map[label_sem_class[idx_to_fill].astype('int')]

        # fill the missing instance results
        for sem_id in np.unique(new_sem_label):
            sem_id_local_map = learning_map_doc[sem_id]
            if sem_id_local_map < 1 or sem_id_local_map > 8:  # stuff class
                valid_ind = np.argwhere((new_sem_label == sem_id) & (new_inst_label == -1))[:, 0]
                new_inst_label[valid_ind] = sem_id
            else:
                # things class that we do not track - give them some ID.
                valid_ind = np.argwhere((new_sem_label == sem_id) & (new_inst_label == -1))[:, 0]
                new_inst_label[valid_ind] = 0
                #new_sem_label[valid_ind] = 0

                if len(valid_ind) > 2:
                    hierarchical_cluster = AgglomerativeClustering(n_clusters=None,
                                                                   distance_threshold=1.5,
                                                                   metric='euclidean',
                                                                   linkage='complete')
                    labels = hierarchical_cluster.fit_predict(points[valid_ind, :3])

                    # draw_scenes_raw(points[valid_ind], title="just painted")
                    for l in np.unique(labels):
                        if l == -1:
                            continue

                        ind = valid_ind[np.where(labels == l)]

                                # if sem_id_local_map==1: # trunks
                                #    draw_scenes(points[valid_ind], title=" all points  =" + str(l))
                                #    draw_scenes(points[ind],title = " instance =" +str(l))

                        if len(ind) >= 25:
                            # draw_scenes_raw(points[ind], title="just painted")
                            new_inst_label[ind] = sem_id
                            new_id += 1
                        else:
                            new_inst_label[ind] = 0
                            new_sem_label[ind] = 0
                else:
                    new_inst_label[valid_ind] = 0
                    new_sem_label[valid_ind] = 0


        assert(len(np.where(new_inst_label == -1)[0]) == 0)
        assert (len(np.where(new_sem_label == -1)[0]) == 0)

        # save results

        # write instances to label file which is binary
        new_inst_label = new_inst_label.astype(np.int32)
        new_preds = np.left_shift(new_inst_label, 16)

        new_sem_label = new_sem_label.astype(np.int32)
        new_preds = np.bitwise_or(new_preds, new_sem_label)


        filename = '{}/{}/{:02d}/predictions/{:06d}.label'.format(save_path, 'sequences', args.sequences, idx)
        new_preds.tofile(filename)

        print("\r", end='')
    # END LOOP OVER FRAMES


    #end scans loop

    print("DONE! ")

    filename =os.path.join(os.path.  dirname(save_path),"trackID_to_labelID.txt")

    # remove old version of this file
    try:
        os.remove(filename)
    except OSError:
        pass

    #'{}/{}/{:02d}/predictions/{:06d}.label'.format(save_path, 'sequences', sequence, idx)
    with open(filename, 'a') as the_file:
        txt = str('cls_group_ind_in_mem') + " , " + str('trk_id') + " , " + str('label_id')
        the_file.write(txt + '\n')
        for cls_group_ind_in_mem in range(3):
            for ii in range( len(IDMemory[cls_group_ind_in_mem].trk_id)):
                txt = str(cls_group_ind_in_mem) + " , " + str(IDMemory[cls_group_ind_in_mem].trk_id[ii]) + " , " + str(IDMemory[cls_group_ind_in_mem].label_id[ii])
                the_file.write(txt + '\n')

if __name__ == '__main__':
    args = parse_args()
    main(args)
