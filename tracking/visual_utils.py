import vispy
from vispy.scene import visuals, SceneCanvas
from box import Box3D


class LaserScanVis:
  #Class that creates and handles a visualizer for a pointcloud
  def __init__(self, points, sem_class, inst_class,cfg,frameNum):

    self.points = points
    self.sem_class = sem_class
    self.inst_class = inst_class
    self.frameNum = frameNum

    # make lut
    # make semantic colors
    sem_color_dict = cfg["color_map"]
    max_sem_key = 0
    for key, data in sem_color_dict.items():
      if key + 1 > max_sem_key:
        max_sem_key = key + 1

    self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
    for key, value in sem_color_dict.items():
      self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

    # make instance colors
    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    # force zero to a gray-ish color
    self.inst_color_lut[0] = np.full((3), 0.5) # 0.1

    self.reset()
    #self.update_scan()

  def reset(self):
    # Reset.
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # figure 1 #  results off 4dstop on the upsample data
    self.figure1 = SceneCanvas(keys='interactive',show=True)
    # interface (n next, b back, q quit, very simple)
    #self.figure1.events.key_press.connect(self.key_press)
    self.figure1.events.draw.connect(self.draw)

    # Create 2 ViewBoxes
    self.track_box_vb = vispy.scene.widgets.ViewBox(border_color='white', parent=self.figure1.scene)

    # Put viewboxes in a grid
    #self.grid = self.figure1.central_widget.add_grid()
    #self.grid.padding = 6
    #self.grid.add_widget(self.track_box_vb, 0, 0)


    # Assign cameras
    self.track_box_vb.camera = 'turntable'#fly' #'turntable'  #dict_keys([None, \'base\', \'panzoom\', \'perspective\', \'turntable\', \'fly\', \'arcball\'])'

    # Add a Node to the scene for this ViewBox.
    self.track_boxes_data_handler = visuals.Markers()
    self.track_box_vb.add(self.track_boxes_data_handler)

    # add titles
    _ = visuals.Text("tracker", pos=(100, 15), font_size=14, color='white', parent=self.track_box_vb)

    #  show 3d axis for indicating coordinate system orientation. Axes are x=red, y=green, z=blue
    visuals.XYZAxis(parent=self.track_box_vb.scene)

  def update_scan(self):

    #  update figure1
    #pred_boxes, pred_TrackedID, pred_ClassID ,_= self.parse_boxes(self.pred_tracking_box_files_names[self.offset])
    #self.clean_draw_boxes()
    #self.draw_boxes(pred_boxes,pred_TrackedID,pred_ClassID)
    self.track_boxes_data_handler.set_data(self.points[:, :3],
                           face_color='white',#'#self.inst_color_lut[self.inst_class],
                           edge_color='white',#self.inst_color_lut[self.inst_class],
                           size=1)

    # then change names
    title = "scan " + str(self.frameNum)
    self.figure1.title = 'My Tracker : '  + title

  # interface
  def key_press(self, event):
    self.figure1.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= self.total:
        self.offset = 0
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      if self.offset < 0:
        self.offset = self.total - 1
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()

  def draw(self, event):
    if self.figure1.events.key_press.blocked():
      self.figure1.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.figure1.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()

  def draw_boxes(self,boxes_list,pred_TrackedID,pred_ClassID, color='red'):
      # input boxes_list is np.array (number_of_boxes X 7)
      local_boxes_data=[]
      local_text_data =[]
      for i, box in enumerate(boxes_list):
          # (float(h), float(w), float(l), float(x), float(y), float(z),float(theta))
          height,width,length,cx, cy, cz,yaw_in_rad = box
          #cx, cy, cz, length, width, height, yaw_in_rad = box

          #yaw_in_degree = (yaw_in_rad / np.pi) * 180

          box1 = vispy.scene.visuals.Box(width=width, height=height, depth=length,color=None,edge_color=color,
                                         width_segments=1,height_segments=1,depth_segments=1)
          # Define a scale and translate transformation :
          transform_ = vispy.visuals.transforms.MatrixTransform()
          transform_.rotate(90, axis=(0, 0, 1))  # detetion angles are rotated. The angle of rotation, in degrees.
          transform_.translate((cx, cy, cz))  # without this box center = axis origin
          box1.transform = transform_

          local_boxes_data.append(box1)

          if self.ShowId:
            # define the text
            info = "ID: " + str(pred_TrackedID[i]) + F"\n" + pred_ClassID[i]
            #info = "ind: " + str(i)
            # point=(bb[0], bb[1] - bb[4] / 2, bb[2] + bb[5] / 2),
            cond1 = self.pred_trackID_debug==None
            cond2 = (pred_TrackedID[i] in self.pred_trackID_debug) if (self.pred_trackID_debug is not None) else False
            if cond1 or cond2:
                t1 = visuals.Text(info, pos=(cx, cy - width / 2, cz + height / 2), font_size=1000, color='white')
                local_text_data.append(t1)

      for b in local_boxes_data:
          b.parent = self.pred_box_vb.scene
          # self.box_vb.add(b)
      if self.ShowId:
          for t in local_text_data:
              t.parent = self.pred_box_vb.scene
              # self.box_vb.add(t)

      if len(self.global_pred_boxes_data)==0:
          self.global_pred_boxes_data = local_boxes_data
      else:
          self.global_pred_boxes_data = self.global_pred_boxes_data.append(local_boxes_data)
      del local_boxes_data
      if self.ShowId:

          if len(self.pred_boxes_text_data) == 0:
              self.pred_boxes_text_data = local_text_data
          else:
              self.pred_boxes_text_data = self.pred_boxes_text_data.append(local_text_data)
          del local_text_data

  def clean_draw_boxes(self):
      if len(self.global_pred_boxes_data) !=0:
          for rb in self.global_pred_boxes_data:
              rb.parent = None
              del rb
          del self.global_pred_boxes_data
          self.global_pred_boxes_data = []

      if self.ShowId:
          if len(self.pred_boxes_text_data)!=0:
              for rt in self.pred_boxes_text_data:
                  rt.parent = None
                  del rt
              del self.pred_boxes_text_data
              self.pred_boxes_text_data = []





"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np


"""
box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

# https://matplotlib.org/stable/tutorials/colors/colormaps.html
N_class = 10
cmap = matplotlib.colormaps['hsv'].resampled(N_class + 1)
box_colormap = cmap(range(N_class +1 ))[:,:-1]

gt_box_colormap = box_colormap * 0.5
gt_box_colormap=gt_box_colormap.astype(np.float)# darker
"""


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True,title=None):
    # assumes boxes is list of the class Box3D
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    itle = "scan " + str(0)
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name = title)

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis =  draw_box(vis,
                        gt_boxes=gt_boxes,
                        color_map=None,
                        color=(0, 0, 1),
                        ref_labels=None,
                        score= None)

    if ref_boxes is not None:
        vis =  draw_box(vis,
                        gt_boxes=ref_boxes,
                        color_map=None,
                        color=(0, 1, 0),
                        ref_labels=ref_labels,
                        score= ref_scores)

    vis.run()
    vis.destroy_window()

def draw_scenes_raw(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True,title=None):
    # assumes box in the format [[h,w,l,x,y,z,theta],...]

    if gt_boxes is not None:
        gt_boxes_list =[]
        for gt_box in gt_boxes:
            gt_boxes_list.append(Box3D.array2bbox_raw(gt_box))
    else:
        gt_boxes_list=None

    if ref_boxes is not None:
        ref_boxes_list =[]
        for ref_box in ref_boxes:
            ref_boxes_list.append(Box3D.array2bbox_raw(ref_box))
    else:
        ref_boxes_list=None

    draw_scenes(points, gt_boxes=gt_boxes_list, ref_boxes=ref_boxes_list, ref_labels=ref_labels, ref_scores=ref_scores, point_colors=point_colors,
                draw_origin=draw_origin,title=title)

def translate_boxes_to_open3d_instance(gt_box):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_box[0:3]
    lwh = gt_box[3:6]
    axis_angles = np.array([0, 0, gt_box[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes,color_map, color=(0, 1, 0),ref_labels=None, score=None):

    for i,gt_box in enumerate(gt_boxes):
        b  = np.array((gt_box.x, gt_box.y,gt_box.z ,gt_box.l,gt_box.w,gt_box.h,gt_box.ry))
        line_set, box3d = translate_boxes_to_open3d_instance(np.squeeze(b))
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(color_map[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

