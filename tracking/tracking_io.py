import os
import numpy as np
import yaml
from tracking.box_tracker import get_detections
from AB3DMOT_libs.io import save_results
def getKalmanDebugFilePath(ParentFilePath,template,debug_id):
    template_ends_with =str(debug_id) +'.txt'
    for root, dirs, files in os.walk(os.path.expanduser(ParentFilePath)):
        for file in files:
            if template.lower()  in file.lower() and file.lower().endswith(template_ends_with):
                return os.path.join(root, file)
    return "" # not found

def getKalmanDebugFilesPath(path_to_dumps,debug_id):

    debug_kalman_predict_file  = getKalmanDebugFilePath(path_to_dumps,'kalman_predict_',debug_id)
    debug_kalman_update_file = getKalmanDebugFilePath(path_to_dumps, 'kalman_update_', debug_id)
    debug_kalman_output_file = getKalmanDebugFilePath(path_to_dumps, 'kalman_output_', debug_id)
    debug_kalman_measurement_file = getKalmanDebugFilePath(path_to_dumps, 'kalman_measurement_', debug_id)
    debug_kalman_P_predict_file  = getKalmanDebugFilePath(path_to_dumps, 'kalman_covP_predict_', debug_id)
    debug_kalman_P_update_file = getKalmanDebugFilePath(path_to_dumps, 'kalman_covP_update_', debug_id)
    debug_kalman_innovation_file = getKalmanDebugFilePath(path_to_dumps, 'kalman_innovation_update_', debug_id)

    if  not os.path.exists(debug_kalman_predict_file):
        raise ValueError('non existing file  : ' + debug_kalman_predict_file)
    if  not os.path.exists(debug_kalman_update_file):
        raise ValueError('non existing file  : ' + debug_kalman_update_file)
    if not os.path.exists(debug_kalman_output_file):
        raise ValueError('non existing file  : ' + debug_kalman_output_file)
    if not os.path.exists(debug_kalman_measurement_file):
        raise ValueError('non existing file  : ' + debug_kalman_measurement_file)
    if not os.path.exists(debug_kalman_P_predict_file):
        raise ValueError('non existing file  : ' + debug_kalman_P_predict_file)
    if not os.path.exists(debug_kalman_P_update_file):
        raise ValueError('non existing file  : ' + debug_kalman_P_update_file)
    if not os.path.exists(debug_kalman_innovation_file):
        raise ValueError('non existing file  : ' + debug_kalman_innovation_file)

    return debug_kalman_predict_file,debug_kalman_update_file,debug_kalman_output_file,debug_kalman_measurement_file,debug_kalman_P_predict_file,debug_kalman_P_update_file,debug_kalman_innovation_file

def getGTScanNamesAndLabelNames(path_to_dataset,sequence_int):

    point_paths = os.path.join(path_to_dataset, "sequences", '{0:02d}'.format(sequence_int), "velodyne")
    point_names = sorted(
        [os.path.join(point_paths, fn) for fn in os.listdir(point_paths) if fn.endswith(".bin")])

    labels_paths = os.path.join(path_to_dataset, "sequences", '{0:02d}'.format(sequence_int), "labels")
    labels_names = sorted([os.path.join(labels_paths, fn) for fn in os.listdir(labels_paths) if fn.endswith(".label")])

    return point_names,labels_names


def parseStateX(filepath):
    #  FrameNum , (state x dimension 10:) x, y, z, theta, l, w, h, dx, dy, dz
    #  constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz
    # removing the new line characters

    y_axis_data = [] # list of lists
    x_axis_data =[]


    if  os.path.exists(filepath):
        with open(filepath) as f:
            lines = [line.rstrip() for line in f]
            for line in lines:
                floats = [float(x) for x in line.split()]
                assert(len(floats)==11) # frameNum + state vector with len 10
                frame_idx = int(floats[0])
                state_x =  floats[1:]
                x_axis_data.append(frame_idx)
                y_axis_data = np.vstack((y_axis_data, state_x)) if len(y_axis_data) != 0 else np.array(state_x, ndmin=2)
    return y_axis_data, x_axis_data

def parseMeasurement(filepath):
    # frameNum,  np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
    y_axis_data = [] # list of lists
    x_axis_data =[]
    if os.path.exists(filepath):
        with open(filepath) as f:
            lines = [line.rstrip() for line in f]
            for line in lines:
                floats = [float(x) for x in line.split()]
                assert(len(floats)==8) # frameNum + state vector with len 7
                frame_idx = int(floats[0])
                state_x =  floats[1:]
                x_axis_data.append(frame_idx)
                y_axis_data = np.vstack((y_axis_data, state_x)) if len(y_axis_data) != 0 else np.array(state_x, ndmin=2)

    return y_axis_data, x_axis_data

def GenerateGTfileWithClassMask(path_to_dataset,GTParentFile,seq_name_int):

    if not os.path.exists(GTParentFile):
        os.makedirs(GTParentFile)

    config_data = yaml.safe_load(open(os.path.join(path_to_dataset,'semantic-kitti.yaml'), 'r'))
    det_id2str=config_data['labels']
    classed_as_string = {value for key, value in config_data['labels'].items()}

    #det_id2str = {key: config_data['labels'][value] for key, value in config_data['learning_map_inv'].items()}


    point_names, label_names= getGTScanNamesAndLabelNames(path_to_dataset,seq_name_int)
    assert(len(point_names)==len(label_names))

    Nframe = len(point_names)
    for idx in range(Nframe):

        point_file = point_names[idx]
        label_file = label_names[idx]

        points = open_scan(point_file)
        gt_seg_label, gt_inst_label = open_label(label_file)

        save_gt_file = os.path.join(GTParentFile, '%06d.txt' % idx)
        save_gt_file = open(save_gt_file, 'w')

        for t in classed_as_string:
            wanted_obj_class_id  = [id for id, str in det_id2str.items() if str == t]
            mask = (gt_seg_label == wanted_obj_class_id)  # if input is from 4DStop, then car class is 1, else car is 10
            dets_frame = get_detections(points, gt_inst_label * mask, gt_seg_label,np.ones(points.shape))

            # draw_scenes(points=obj_points, title='get_detections_from_segmentation: obj_points')
            # draw_scenes_raw(points=points, gt_boxes=dets_frame['dets'], title='get_detections_from_segmentation : detections')


            for j in range(len(dets_frame['dets'])):
                h, w, l, x, y, z, theta = dets_frame['dets'][j]
                classID, TrackID = dets_frame['info'][j][1], dets_frame['info'][j][2]
                # expecting  format [bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry](1x7) ,[trk.id](1x1), trk.info[1x8])

                result_tmp = np.array((h,w,l,x,y,z,theta,TrackID,np.nan,classID,np.nan,np.nan,np.nan,np.nan,np.nan))


                save_results(result_tmp, save_gt_file, None, det_id2str, idx, -1000000)
        save_gt_file.close()

def parse_P(filepath):
    #  FrameNum , (cov P dimension 10x10)
    y_axis_data = []  # list of lists
    x_axis_data = []

    if os.path.exists(filepath):
        with open(filepath) as f:
            lines = [line.rstrip() for line in f]
            for line in lines:
                floats = [float(x) for x in line.split()]
                assert (len(floats) == 1+10*10)  # frameNum + matrix P with len 10x10
                frame_idx = int(floats[0])
                P_mat = np.array(floats[1:]).reshape(10,10)
                P_diagonal = np.diag(P_mat)#P_flat # your_list[start:end:jump]

                x_axis_data.append(frame_idx)
                y_axis_data = np.vstack((y_axis_data, P_diagonal)) if len(y_axis_data) != 0 else np.array(P_diagonal, ndmin=2)
    return y_axis_data, x_axis_data

def open_GTfiles(ParentFilename,debug_class_str,debug_Trackid):
    """ Open raw scan and fill in attributes
    """

    if not os.path.exists(ParentFilename):
     raise RuntimeError("file path not exist  " + ParentFilename )

    gt_files_names = sorted(
        [os.path.join(ParentFilename, fn) for fn in os.listdir(ParentFilename) if fn.endswith(".txt")])

    frameNums = []    # get xyz
    classIDs = []
    trackIDs = []
    box_data = []
    for filename in gt_files_names:
        frameNum = int(os.path.split(os.path.splitext(filename)[0])[1])
        array_of_boxes, array_of_TrackedID, array_of_ClassID,array_of_DetectionScore=parse_boxes(filename)

        # array_of_boxes:

        find_class_idx  = np.where(array_of_ClassID==debug_class_str)[0]
        if len(find_class_idx)!=0:

            # found
            find_trackID_idx = np.where(array_of_TrackedID==debug_Trackid)[0]
            if len(find_trackID_idx)!= 0:

                idx = np.intersect1d(find_class_idx,find_trackID_idx)
                if len(idx)!=0:

                    if len(idx)>1:
                        raise RuntimeError("found too many matches")

                    # found! fill data
                    frameNums.append(frameNum)  # get xyz
                    classIDs.append(array_of_ClassID[idx])
                    trackIDs.append(array_of_TrackedID[idx])

                    height,width,length,cx, cy, cz,yaw_in_rad = array_of_boxes[idx,:].T # height,width,length,cx, cy, cz,yaw_in_rad = box
                    local_box_converted =np.array((cx,cy,cz,yaw_in_rad,length,width,height)).T #7X1# [bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h]
                    box_data = np.vstack((box_data, local_box_converted)) if len(box_data) != 0 else local_box_converted  # [bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h]

    return frameNums,classIDs,trackIDs,box_data

def parseGT(GT_parent_file,gt_class_str,gt_trackID):
    # frameNum,  np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
    y_axis_data = [] # list of lists
    x_axis_data =[]
    gt_frameNums,gt_classIDs,gt_trackIDs,gt_box_data = open_GTfiles(GT_parent_file,gt_class_str,gt_trackID)

    x_axis_data = gt_frameNums
    y_axis_data = gt_box_data

    return y_axis_data, x_axis_data

def open_scan(filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in '.bin'):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    # put in attribute
    #points = scan[:, 0:3]    # get xyz
    #remissions = scan[:, 3]  # get remission
    return scan


def open_label(filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in '.label'):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))

    # set it
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    return sem_label,inst_label


def parse_boxes(filename):
    # expected file in kitti format
    # return
    #    :param boxes: (array(N,7)), 3D boxes
    #    :param TrackedID: list(N,), the ID of each box
    #    :param ClassID: (list(N,)), a list of str, the infos of boxes to show
    #    #print("parse_boxes : filename = ",os.path.basename(filename))

    file = open(filename)
    array_of_boxes= np.empty(0, float)
    array_of_TrackedID = np.empty(0, int)
    array_of_ClassID = np.empty(0, str)
    array_of_DetectionScore = np.empty(0, float)
    for line in file:
        strclassID, _, _, _, _, _, _, _, h, w, l, x, y, z, theta, detectionScore, trackedID= line.strip().split() # box3d in the format of h, w, l, x, y, z, theta in camera coordinate
        bb = np.array((float(h), float(w), float(l), float(x), float(y), float(z),float(theta)))
        array_of_boxes = np.vstack((array_of_boxes, bb)) if len(array_of_boxes) != 0 else bb.reshape((1,7))
        array_of_TrackedID = np.append(array_of_TrackedID, int(trackedID))
        array_of_ClassID = np.append(array_of_ClassID, strclassID)
        array_of_DetectionScore= np.append(array_of_DetectionScore,float(detectionScore))
    return  array_of_boxes, array_of_TrackedID, array_of_ClassID,array_of_DetectionScore


def get_filesnames(parent_path,extension):

    if parent_path is not None and os.path.isdir(parent_path):
        filesnames =[]
        print("reading filenames from " , parent_path)
        # populate the filenames
        for root, dirs, files in os.walk(os.path.expanduser(parent_path)):
            for file in files:
                if file.lower().endswith(extension):
                    filesnames.append(os.path.join(root, file))
        filesnames.sort()
    else:
        filesnames = None
    return filesnames

def load_pred_volume(filename,voxel_dims):
    labels = np.fromfile(filename, dtype=np.uint16).astype(np.float32)
    labels = labels.reshape(voxel_dims)
    return labels # labels != 0  # labels

def loadJS3C (filename,CFG):

    # voxel parameters
    min_range = 2.5
    max_range = 70
    future_scans = 70
    min_extent = [0, -25.6, -2]
    max_extent = [51.2, 25.6, 4.4]
    voxel_size = 0.2
    voxel_dims = (256, 256, 32)

    min_x = min_extent[0]
    min_y = min_extent[1]
    min_z = min_extent[2]

    max_x = max_extent[0]
    max_y = max_extent[1]
    max_z = max_extent[2]

    ## map and colors
    color_dict_bgr = CFG["color_map"]
    learning_map = dict(CFG["learning_map"])

    mapping = {k: v for k, v in zip(learning_map.keys(), range(len(learning_map)))}

    semantic_colors = np.asarray(
        [list(color_dict_bgr[k]) for k in learning_map.keys()], np.uint8
    )
    # BGR -> RGB
    semantic_colors = semantic_colors[..., ::-1]

    pred_voxels = load_pred_volume(filename,voxel_dims)
    dtype = np.float32
    deltas = np.asarray([voxel_size, voxel_size, voxel_size], dtype=dtype)
    pred_voxel_coords = np.mgrid[[slice(x) for x in pred_voxels.shape]].astype(dtype)
    pred_voxel_coords = np.moveaxis(pred_voxel_coords, 0, 3)
    pred_voxel_coords *= deltas

    # nirit fix shift bug according to  https://github.com/PRBonn/semantic-kitti-api/issues/46
    sizex_ = np.ceil((max_x - min_x) / voxel_size);
    sizey_ = np.ceil((max_y - min_y) / voxel_size);
    sizez_ = np.ceil((max_z - min_z) / voxel_size);

    shiftx = min_x - 0.5 * (sizex_ * voxel_size - (max_x - min_x))
    shifty = min_y - 0.5 * (sizey_ * voxel_size - (max_y - min_y))
    shiftz = min_z - 0.5 * (sizez_ * voxel_size - (max_z - min_z))

    shift = 0.5 * voxel_size - np.array((shiftx, shifty, shiftz))

    pred_voxel_coords = pred_voxel_coords - shift
    pred_voxel_coords = pred_voxel_coords[pred_voxels != 0]


    color_dict_bgr = CFG["color_map"]
    learning_map = dict(CFG["learning_map"])

    mapping = {k: v for k, v in zip(learning_map.keys(), range(len(learning_map)))}

    voxel_label = np.vectorize(mapping.get, otypes=[np.int16])(pred_voxels)
    colormap = semantic_colors[voxel_label]
    pred_voxel_colors = colormap[pred_voxels != 0, :] / 255

    # do not show black results
    pred_zero_color_idx = [];
    pred_color_to_keep_idx = []
    for idx, c in enumerate(pred_voxel_colors):
        if c[0] == c[1] == c[1] == 0:
            pred_zero_color_idx.append(idx)
        else:
            pred_color_to_keep_idx.append(idx)

    pred_voxel_coords = pred_voxel_coords[pred_color_to_keep_idx, :]
    pred_voxel_colors = pred_voxel_colors[pred_color_to_keep_idx, :]

    # nirit to do this more auto , from yaml
    things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']

    # bgr
    car_color = [245, 150, 100]  # 10
    truck_color = [180, 30, 80]  # 18
    bicycle_color = [245, 230, 100]  # 11
    motorcycle_color = [150, 60, 30]  # 15
    other_vehicle_color = [255, 0, 0]  # 20
    person_color = [30, 30, 255]  # 30
    bicyclist_color = [200, 40, 255]  # 31
    motorcyclist_color = [90, 30, 150]  # 32

    idx_to_keep = []
    seg_label = []
    inst_label = []
    for idx, c in enumerate(pred_voxel_colors):
        to_uint8 = c * 255;

        # swap axis of color
        tmp = to_uint8[2]
        to_uint8[2] = to_uint8[0]
        to_uint8[0] = tmp

        if all(to_uint8[:] == car_color[:]):
            cur_seg_label = 10

        if all(to_uint8[:] == truck_color[:]):
            cur_seg_label = 18

        if all(to_uint8[:] == bicycle_color[:]):
            cur_seg_label = 11

        if all(to_uint8[:] == motorcycle_color[:]):
            cur_seg_label = 15

        if all(to_uint8[:] == other_vehicle_color[:]):
            cur_seg_label = 20

        if all(to_uint8[:] == person_color[:]):
            cur_seg_label = 30

        if all(to_uint8[:] == bicyclist_color[:]):
            cur_seg_label = 31

        if all(to_uint8[:] == motorcyclist_color[:]):
            cur_seg_label = 32

        if all(to_uint8[:] == car_color[:]) or \
                all(to_uint8[:] == truck_color[:]) or \
                all(to_uint8[:] == bicycle_color[:]) or \
                all(to_uint8[:] == motorcycle_color[:]) or \
                all(to_uint8[:] == other_vehicle_color[:]) or \
                all(to_uint8[:] == person_color[:]) or \
                all(to_uint8[:] == bicyclist_color[:]) or \
                all(to_uint8[:] == motorcyclist_color[:]) \
                :
            idx_to_keep.append(idx)
            seg_label.append(cur_seg_label)

    return pred_voxel_coords,seg_label