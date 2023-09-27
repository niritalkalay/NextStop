import os
import numpy as np
import yaml
from tracking.tracking_io import parse_boxes
from dataclasses import dataclass

path_to_dataset = '/media/nirit/mugiwara/datasets/SemanticKitti/'
yaml_path = os.path.join(path_to_dataset, 'semantic-kitti.yaml')

#GT
#path_to_GT = '/media/nirit/mugiwara/code/4D-StOP/media/nirit/mugiwara/code/4D-StOP/test/Log_2022-06-13_17-33-24_importance_None_str1_bigpug_2_current_chkp/AB3DMOT_tracker/debug_kalman/GT/08'

# orig
#path_to_GT='/media/nirit/mugiwara/code/4D-StOP/media/nirit/mugiwara/code/4D-StOP/test/Log_2022-06-13_17-33-24_importance_None_str1_bigpug_2_current_chkp/AB3DMOT_tracker/predictions/trk_withid_0_AB3DMOT_orig/sequences08'

# latest run
path_to_GT='/media/nirit/mugiwara/code/4D-StOP/media/nirit/mugiwara/code/4D-StOP/test/Log_2022-06-13_17-33-24_importance_None_str1_bigpug_2_current_chkp/AB3DMOT_tracker/predictions/trk_withid_0/sequences08'

# cars
#path_to_GT='/media/nirit/mugiwara/code/4D-StOP/media/nirit/mugiwara/code/4D-StOP/test/Log_2022-06-13_17-33-24_importance_None_str1_bigpug_2_current_chkp/AB3DMOT_tracker/ours_with_sc/predictions/trk_withid_0/sequences08'


gt_names = []
for root, dirs, files in os.walk(os.path.expanduser(path_to_GT)):
    for file in files:
        if file.lower().endswith('.txt'):
            gt_names.append(os.path.join(root, file))
gt_names.sort()
print('len(gt_names)=', len(gt_names))

def main():
 # once GT Folder with GTtrackID has been created
 # use this file here to locate information about the length of eacc id.
 nframe = len(gt_names)
 config_data = yaml.safe_load(open(yaml_path, 'r'))

 @dataclass
 class info:
     id: int
     frames: list()
     box_volume: list()


 results = {value: [] for key, value in config_data['labels'].items()}

 for filename in gt_names:
    frameNum = int(os.path.split(os.path.splitext(filename)[0])[1])

    array_of_boxes, array_of_TrackedID, array_of_ClassID, array_of_DetectionScore = parse_boxes(filename)

    for i,class_str in enumerate(array_of_ClassID):
        box = array_of_boxes[i, :]
        vol = box[0] * box[1] * box[2]

        if len(results[class_str])==0:
            results[class_str] = [(info(id=array_of_TrackedID[i], frames=[frameNum],box_volume =[vol]))]
        else:
            found = False
            for l in results[class_str]:
                if l.id == array_of_TrackedID[i]:
                    l.frames.append(frameNum)
                    l.box_volume.append(vol)
                    found=True
                    break
            if not found:
                results[class_str].append((info(id=array_of_TrackedID[i], frames=[frameNum],box_volume =[vol])))

 vehicles_str = ["car", "moving-car", "bus", "moving-bus", "truck", "moving-truck", "other-vehicle","moving-other-vehicle"]
 bikes_str = ["bicycle", "bicyclist", "moving-bicyclist", "motorcycle", "motorcyclist", "moving-motorcyclist"]
 Pedestrian_str = ["person", "moving-person"]



 #debug_idx_list =[16149,16354,17102,17141,17181,17219,17238,17383,17164,17468,17510,17544,17574,17587,17621,17728,17761]
 #debug_idx_list = [24]
 debug_idx_list = None
 # print stats

 short_frame_counter=0
 for key, value in results.items():
     if key in vehicles_str or key in bikes_str or  key in Pedestrian_str:
         #id_list=[]
         #frame_count =[]
         if len(value) !=0:
             print("class ", key, " has " , len(value)," trackID :")
         total_ids_count=0
         for v in value:
             #id_list = np.vstack((id_list, v[0])) if len(id_list) != 0 else np.array(v[0], ndmin=2)
             #frame_count =np.vstack((frame_count, len(v[1:]))) if len(id_list) != 0 else np.array(len(v[1:]), ndmin=2)
             if debug_idx_list is not None:
                 if v.id in debug_idx_list:
                     print("    id = {0}, total_frames={1} : start {2} , end {3} , mean volume : {4}".format(v.id,len(v.frames ),np.min(v.frames),
                                                                                          np.max(v.frames),np.mean(v.box_volume) ))
             else:
                 print("    id = {0}, total_frames={1} : start {2} , end {3} , mean volume : {4}".format(v.id, len(v.frames), np.min(v.frames),
                                                                                  np.max(v.frames),np.mean(v.box_volume)))

                 if (len(v.frames))<10:
                     short_frame_counter = short_frame_counter + 1



 print("short_frame_counter=",short_frame_counter)

if __name__ == "__main__":
    main()
    print('end of main!')