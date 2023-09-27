# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np, os
from tracking.box import Box3D # nirit original: from AB3DMOT_libs.box import Box3D
from AB3DMOT_libs.matching import data_association,refine_detection
from AB3DMOT_libs.kalman_filter import KF
from Xinshuo_PyToolbox_master.xinshuo_miscellaneous import print_log
from tracking.visual_utils import draw_scenes,draw_scenes_raw

np.set_printoptions(suppress=True, precision=3)

# A Baseline of 3D Multi-Object Tracking
class AB3DMOT(object):			  	
	def __init__(self, cfg, cat,debug_path, calib=None, oxts=None, log=None, ID_init=0):

		# vis and log purposes
		self.vis = cfg.vis # todo
		self.log = log

		# counter
		self.ActiveTrackers = []
		self.CandidatesTrackers = []
		self.frame_count = 0
		self.ID_count = [ID_init]
		self.id_now_output = []

		# config
		self.cat = cat
		self.ego_com = cfg.ego_com 			# ego motion compensation
		self.calib = calib
		self.oxts = oxts
		self.affi_process = cfg.affi_pro	# post-processing affinity
		self.get_param(cfg, cat)
		self.print_param()

		# debug
		#self.debug_id = 29
		self.debug_id = None
		self.debug_path = debug_path
		self.debug_kalman_path = None
		self.debug_kalman_predict_file = None
		self.debug_kalman_update_file = None
		self.debug_kalman_output_file = None
		self.debug_kalman_measurement_file =None
		self.debug_kalman_cov_P_predict_file = None
		self.debug_kalman_cov_P_update_file = None

		if self.debug_id is not None:
			if not os.path.exists(self.debug_path):
				os.makedirs(self.debug_path)

			self.debug_kalman_path  = os.path.join(self.debug_path,"debug_kalman")
			if not os.path.exists(self.debug_kalman_path):
				os.makedirs(self.debug_kalman_path)

			self.debug_kalman_predict_file = os.path.join(self.debug_kalman_path ,'kalman_predict_Trackid_'+str(self.debug_id)+'.txt')
			self.debug_kalman_update_file = os.path.join(self.debug_kalman_path , 'kalman_update_Trackid_'+str(self.debug_id)+'.txt')
			self.debug_kalman_output_file  = os.path.join(self.debug_kalman_path , 'kalman_output_Trackid_'+str(self.debug_id)+'.txt')
			self.debug_kalman_measurement_file = os.path.join(self.debug_kalman_path,'kalman_measurement_Trackid_' + str(self.debug_id) + '.txt')
			self.debug_kalman_cov_P_predict_file = os.path.join(self.debug_kalman_path,'kalman_covP_predict_Trackid_' + str(self.debug_id) + '.txt')
			self.debug_kalman_cov_P_update_file = os.path.join(self.debug_kalman_path,'kalman_covP_update_Trackid_' + str(self.debug_id) + '.txt')
			self.debug_kalman_Innovation_update_file = os.path.join(self.debug_kalman_path,
															   'kalman_Innovation_update_Trackid_' + str(
																   self.debug_id) + '.txt')


			# remove old files
			if os.path.exists(self.debug_kalman_predict_file):
				os.remove(self.debug_kalman_predict_file)
			if os.path.exists(self.debug_kalman_update_file):
				os.remove(self.debug_kalman_update_file)
			if os.path.exists(self.debug_kalman_output_file):
				os.remove(self.debug_kalman_output_file)
			if os.path.exists(self.debug_kalman_measurement_file):
				os.remove(self.debug_kalman_measurement_file)
			if os.path.exists(self.debug_kalman_cov_P_predict_file):
				os.remove(self.debug_kalman_cov_P_predict_file)
			if os.path.exists(self.debug_kalman_cov_P_update_file):
				os.remove(self.debug_kalman_cov_P_update_file)
			if os.path.exists(self.debug_kalman_Innovation_update_file):
				os.remove(self.debug_kalman_Innovation_update_file)

	@staticmethod
	def saveP(frameNum,P,file_name):
		# state P dimension 10x10:
		# constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz

		mylist =  np.append(frameNum,P.ravel())
		mylist_as_string = ["%f " % el for el in mylist]
		text_to_append="".join(mylist_as_string)

		# Open the file in append & read mode ('a+')
		with open(file_name, "a+") as file_object:
			# Move read cursor to the start of file.
			file_object.seek(0)
			# If file is not empty then append '\n'
			data = file_object.read(100)
			if len(data) > 0:
				file_object.write("\n")
			# Append text at the end of file
			file_object.write(text_to_append)

	@staticmethod
	def saveStateX(frameNum,x,file_name):
		# state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
		# constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz

		mylist =  np.append(frameNum,x)
		mylist_as_string = ["%f " % el for el in mylist]
		text_to_append="".join(mylist_as_string)

		# Open the file in append & read mode ('a+')
		with open(file_name, "a+") as file_object:
			# Move read cursor to the start of file.
			file_object.seek(0)
			# If file is not empty then append '\n'
			data = file_object.read(100)
			if len(data) > 0:
				file_object.write("\n")
			# Append text at the end of file
			file_object.write(text_to_append)

	@staticmethod
	def saveMeasurmentZ(frameNum,box,file_name):
		# box format
		#if bbox.c is None:
		#	return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
		#else:
		#	return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h, bbox.c])

		mylist = np.append(frameNum, box)
		mylist_as_string = ["%f " % el for el in mylist]
		text_to_append = "".join(mylist_as_string)

		# Open the file in append & read mode ('a+')
		with open(file_name, "a+") as file_object:
			# Move read cursor to the start of file.
			file_object.seek(0)
			# If file is not empty then append '\n'
			data = file_object.read(100)
			if len(data) > 0:
				file_object.write("\n")
			# Append text at the end of file
			file_object.write(text_to_append)
	def saveInnovation(self,frameNum,y,file_name):
		self.saveMeasurmentZ(frameNum, y, file_name)

	def get_param(self, cfg, cat):
		# get parameters for each dataset

		if cfg.dataset == 'KITTI':
			if cfg.det_name == 'pvrcnn':				# tuned for PV-RCNN detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4 		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
				else: assert False, 'error'
			elif cfg.det_name == 'pointrcnn':			# tuned for PointRCNN detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4 		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
				else: assert False, 'error'
			elif cfg.det_name == 'deprecated':			
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 1, 3, 2		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
				else: assert False, 'error'
			else: assert False, 'error'
		elif cfg.dataset == 'nuScenes':
			if cfg.det_name == 'centerpoint':		# tuned for CenterPoint detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.5, 1, 2
				elif cat == 'Truck': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 2
				elif cat == 'Trailer': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.3, 3, 2
				elif cat == 'Bus': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 2
				elif cat == 'Motorcycle':	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.7, 3, 2
				elif cat == 'Bicycle': 		algm, metric, thres, min_hits, max_age = 'greedy', 'dist_3d',    6, 3, 2
				else: assert False, 'error'
			elif cfg.det_name == 'megvii':			# tuned for Megvii detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.5, 1, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'dist_3d',    2, 1, 2
				elif cat == 'Truck': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.2, 1, 2
				elif cat == 'Trailer': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.2, 3, 2
				elif cat == 'Bus': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.2, 1, 2
				elif cat == 'Motorcycle':	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.8, 3, 2
				elif cat == 'Bicycle': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.6, 3, 2
				else: assert False, 'error'
			elif cfg.det_name == 'deprecated':		
				if cat == 'Car': 			metric, thres, min_hits, max_age = 'dist', 10, 3, 2
				elif cat == 'Pedestrian': 	metric, thres, min_hits, max_age = 'dist',  6, 3, 2	
				elif cat == 'Bicycle': 		metric, thres, min_hits, max_age = 'dist',  6, 3, 2
				elif cat == 'Motorcycle':	metric, thres, min_hits, max_age = 'dist', 10, 3, 2
				elif cat == 'Bus': 			metric, thres, min_hits, max_age = 'dist', 10, 3, 2
				elif cat == 'Trailer': 		metric, thres, min_hits, max_age = 'dist', 10, 3, 2
				elif cat == 'Truck': 		metric, thres, min_hits, max_age = 'dist', 10, 3, 2
				else: assert False, 'error'
			else: assert False, 'error'
		elif cfg.dataset == 'SemanticKITTI':
			if cfg.det_name == '4D-STop':  # tuned for 4D-STop detections
				if cat == 'Car':
					print('Car')
					#algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 6
					#algm, metric, thres, min_hits, max_age = 'hungar', 'PointsSimilarity', 1, 3, 6
					# CAR!
					algm           = 'hungar'
					metric         = 'diou_3d'
					metric_thres_1 = -0.2
					metric_thres_2 = -0.5
					score_thres    = 0.7 #detection thres for high  score,
					min_hits       = 2#3
					max_age        = 2# 3
					death_age      = 10
					refine_thresh  = -0.1 #-0.3 # not in use

					## active tracker params
					Q = np.eye(10) # CAR!
					Q[:3, :3] *= 0.0
					Q[7:, 7:] *= 0.01
					Q[6, 6] = 0.3
					self.Q_active = Q

					R = np.eye(7) # CAR!
					R[3, 3] = 10 ** 4
					R /= 10
					self.R_active = R

					# candidate tracker params # CAR!
					self.Q_candidate = self.Q_active
					self.R_candidate = self.R_active

					# orig values
					P0 = np.eye(10) # CAR!
					P0[7:, 7:] *= 1000.
					P0 *= 10
					self.P0_candidate = P0

				elif cat == 'Pedestrian':
					print('Pedestrian')

					algm = 'hungar'
					metric = 'diou_3d'  # 'dist_3d'
					metric_thres_1 = -0.4  # 4
					metric_thres_2 = -0.7  # 5
					score_thres = 0.8  # 0.78
					min_hits = 3
					max_age = 4
					death_age = 7
					refine_thresh = -0.2



					"""
					
					algm = 'hungar'#'greedy'
					metric =  'dist_3d' #'diou_3d'
					thres = 2.5 #-0.3
					min_hits = 3
					max_age = 3
					death_age = 10
					refine_thresh = -0.#-0.2
					"""
					# kalman params
					## active tracker params

					Q = np.eye(10)  # Pedestrian!
					Q[:3, :3] *= 0.0
					Q[7:, 7:] *= 0.01
					Q[4, 4] = 0.4
					Q[5, 5] = 0.4
					Q[6, 6] = 0.4

					self.Q_active = Q

					R = np.eye(7) # Pedestrian!
					R[3, 3] = 10 ** 4
					R /= 10

					self.R_active = R

					# candidate tracker params
					self.Q_candidate = self.Q_active
					self.R_candidate = self.R_active

					# orig values
					P0 = np.eye(10)  # Pedestrian
					P0[7:, 7:] *= 1000.
					P0 *= 10
					self.P0_candidate = P0

				elif cat == 'Cyclist':
					print('Cyclist')
					algm = 'greedy'#'hungar'
					metric = 'diou_3d' #'dist_3d'
					metric_thres_1 = -0.4#4
					metric_thres_2 = -0.7#5
					score_thres    = 0.8#0.78
					min_hits = 3
					max_age = 4
					death_age = 7
					refine_thresh = -0.1#-0.2

					# kalman params
					## active tracker params
					Q = np.eye(10)  # BIKE!
					Q[:3, :3] *= 0.0
					Q[7:, 7:] *= 0.01
					Q[6, 6] = 0.3
					self.Q_active = Q

					R = np.eye(7) # BIKE
					R[3, 3] = 10 ** 4
					R /= 10
					self.R_active = R

					# candidate tracker params
					self.Q_candidate = self.Q_active
					self.R_candidate = self.R_active

					# orig values
					P0 = np.eye(10) # BIKE
					P0[7:, 7:] *= 1000.
					P0 *= 10
					self.P0_candidate = P0

				else:
					assert False, 'error'


		else: assert False, 'no such dataset'

		# add negative due to it is the cost
		if metric in ['dist_3d', 'dist_2d', 'm_dis', 'PointsSimilarity']:
			metric_thres_1 *= -1
			metric_thres_2 *= -1
		self.algm, self.metric, self.metric_thres_1,self.metric_thres_2,self.score_thres, self.max_age, self.min_hits = \
			algm, metric, metric_thres_1, metric_thres_2, score_thres, max_age, min_hits

		death_age_exists = 'death_age' in locals()# or 'var' in globals()
		if death_age_exists:
			self.death_age = death_age
		else:
			self.death_age = max_age


		# define max/min values for the output affinity matrix
		if self.metric in ['dist_3d', 'dist_2d', 'm_dis']: self.max_sim, self.min_sim = 0.0, -100.
		elif self.metric in ['iou_2d', 'iou_3d']:   	   self.max_sim, self.min_sim = 1.0, 0.0
		elif self.metric in ['giou_2d', 'giou_3d']: 	   self.max_sim, self.min_sim = 1.0, -1.0
		elif self.metric in ['diou_2d', 'diou_3d']:        self.max_sim, self.min_sim = 1.0, -1.0
		elif self.metric in ['PointsSimilarity']:          self.max_sim, self.min_sim = 0.0, -100

		refine_thresh_exists = 'refine_thresh' in locals()# or 'var' in globals()
		if refine_thresh_exists:
			self.refine_thresh = refine_thresh
		else:
			self.refine_thresh = self.min_sim

	def print_param(self):
		print_log('\n\n***************** Parameters for %s *********************' % self.cat, log=self.log, display=False)
		print_log('matching algorithm is %s' % self.algm, log=self.log, display=False)
		print_log('distance metric is %s' % self.metric, log=self.log, display=False)
		print_log('distance threshold1 is %f' % self.metric_thres_1, log=self.log, display=False)
		print_log('distance threshold2 is %f' % self.metric_thres_2, log=self.log, display=False)
		print_log('min hits is %f' % self.min_hits, log=self.log, display=False)
		print_log('max age is %f' % self.max_age, log=self.log, display=False)
		print_log('death age is %f' % self.death_age, log=self.log, display=False)
		print_log('ego motion compensation is %d' % self.ego_com, log=self.log, display=False)

	@staticmethod
	def process_dets(dets):
		# convert each detection into the class Box3D 
		# inputs: 
		# 	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]

		dets_new = []
		for det in dets:
			det_tmp = Box3D.array2bbox_raw(det)
			dets_new.append(det_tmp)

		return dets_new

	@staticmethod
	def within_range(theta):
		# make sure the orientation is within a proper range

		if theta >= np.pi: theta -= np.pi * 2    # make the theta still in the range
		if theta < -np.pi: theta += np.pi * 2

		return theta

	def orientation_correction(self, theta_pre, theta_obs):
		# update orientation in propagated tracks and detected boxes so that they are within 90 degree
		
		# make the theta still in the range
		theta_pre = self.within_range(theta_pre)
		theta_obs = self.within_range(theta_obs)

		# if the angle of two theta is not acute angle, then make it acute
		if  np.pi / 2.0 < abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:
			theta_pre += np.pi       
			theta_pre = self.within_range(theta_pre)

		# now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
		if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
			if theta_obs > 0: theta_pre += np.pi * 2
			else: theta_pre -= np.pi * 2

		return theta_pre, theta_obs

	def KalmanPrediction(self,trks,frameNum,bool_predict_active):
		# get predicted locations from existing tracks

		pred_trks = []
		for t in range(len(trks)):
			# propagate locations
			kf_tmp = trks[t]
			if kf_tmp.id == self.debug_id:
				#print('\n before prediction')
				print(kf_tmp.kf.x.reshape((-1)))
				print('\n current velocity')
				print(kf_tmp.get_velocity())

			if bool_predict_active:
				kf_tmp.kf.predict(Q=self.Q_active)
			else:
				kf_tmp.kf.predict(Q=self.Q_candidate)

			if kf_tmp.id == self.debug_id:
				print('After prediction')
				print(kf_tmp.kf.x.reshape((-1)))
				self.saveStateX(frameNum, kf_tmp.kf.x, file_name=self.debug_kalman_predict_file)
				self.saveP(frameNum, kf_tmp.kf.P, file_name=self.debug_kalman_cov_P_predict_file)

			kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])
			# update statistics
			kf_tmp.time_since_update += 1
			trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
			pred_trks.append(Box3D.array2bbox(trk_tmp))

		return pred_trks

	def KalmanUpdate(self, matched, unmatched_trks, dets, info,trks, frameNum,bool_update_active):
		# update matched trackers with assigned detections

		updated_trks  = trks
		for t, trk in enumerate(trks):
			if t not in unmatched_trks:
				d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
				assert len(d) == 1, 'error'

				# update statistics
				trk.time_since_update = 0  # reset because just updated
				trk.hits += 1

				# update orientation in propagated tracks and detected boxes so that they are within 90 degree
				bbox3d = Box3D.bbox2array(dets[d[0]])
				trk.kf.x[3], bbox3d[3] = self.orientation_correction(trk.kf.x[3], bbox3d[3])


				if self.cat=="Car":
					bbox3d[2] += 0.05
					bbox3d[6] -= 0.1
				elif self.cat=="Cyclist":
					bbox3d[2] -= 0.025 # '(GT -measure) < 0' -> 'measure > gt' with bias, hence need to do 'measure-bias', so  (GT -measure) will be around zero
					bbox3d[6] += 0.0625
				elif self.cat == "Pedestrian":
					bbox3d[2] += 0.028125 # '(GT -measure) < 0' -> 'measure > gt' with bias, hence need to do 'measure-bias', so  (GT -measure) will be around zero
					bbox3d[6] -= 0.1


				if trk.id == self.debug_id:
					print('KalmanUpdate  :   before update candidates')
					print(trk.kf.x.reshape((-1)))
					print('KalmanUpdate : matched measurement')
					print(bbox3d.reshape((-1)))
					# print('uncertainty')
					# print(trk.kf.P)
					# print('measurement noise')
					# print(trk.kf.R)
					self.saveMeasurmentZ(frameNum, bbox3d, file_name=self.debug_kalman_measurement_file)

				# kalman filter update with observation
				if bool_update_active:
					trk.kf.update(bbox3d,R = self.R_active)
				else:
					trk.kf.update(bbox3d, R=self.R_candidate)

				if trk.id == self.debug_id:
					print('KalmanUpdate : after matching')
					print(trk.kf.x.reshape((-1)))
					print('\nKalmanUpdate : current velocity')
					print(trk.get_velocity())

					self.saveStateX(frameNum, trk.kf.x, file_name=self.debug_kalman_update_file)
					self.saveP(frameNum, trk.kf.P, file_name=self.debug_kalman_cov_P_update_file)
					self.saveInnovation(frameNum, trk.kf.y, file_name=self.debug_kalman_Innovation_update_file)

				trk.kf.x[3] = self.within_range(trk.kf.x[3])

				# the current trk class determined with majority.
				# this to avoid errors due to wrong detection classification
				det_class_id = int(info[d, :][0][1])
				trk.count_classes[det_class_id] +=1

				majority_class_ind,majority_count_appearances= trk.get_majority_class()

				trk.info = info[d, :][0]
				trk.info[1] = majority_class_ind  #info[d, 1]           #

				updated_trks[t] = trk

		return updated_trks

	# debug use only
	# else:
	# print('track ID %d is not matched' % trk.id)

	def tracklet_generation(self, dets, info, unmatched_dets,c):
		# create and initialise new trackers for unmatched detections

		# dets = copy.copy(dets)
		new_id_list = list()					# new ID generated for unmatched detections
		for i in unmatched_dets:        			# a scalar of index

			if info[i,6] >= self.score_thres:
				if self.debug_id == self.ID_count[0]:
					print("tracklet_generation : ")

				trk = KF(Box3D.bbox2array(dets[i]), info[i, :], self.ID_count[0],P0=self.P0_candidate) # here original
				self.CandidatesTrackers.append(trk)
				new_id_list.append(trk.id)
				# print('track ID %s has been initialized due to new detection' % trk.id)

				self.ID_count[0] += 1

		return new_id_list

	def candidates_association_and_KalmanUpdate(self,dets,unmatched_dets_ind_input, info,candidates_trks, frame):

		unmatched_dets = [d for ind,d in enumerate(dets) if ind in unmatched_dets_ind_input]
		unmatched_info = [i for ind, i in enumerate(info) if ind in unmatched_dets_ind_input]
		unmatched_info=np.array(unmatched_info)

		matched_to_candidates_trks, unmatched_dets_ind2, unmatched_candidate_trks, cost2, affi2 = \
			data_association(unmatched_dets, candidates_trks, self.metric, self.thres, self.algm, None, points=None)


		# kalman update
		self.CandidatesTrackers = self.KalmanUpdate(matched_to_candidates_trks, unmatched_candidate_trks,
													unmatched_dets,unmatched_info,
													self.CandidatesTrackers,frame,bool_update_active=False)

		# unmatched_trks are false alarm trackers, removed from the candidate list

		unmatched_trks_ids = [candidate_trks.id for i,candidate_trks in enumerate(self.CandidatesTrackers) if i in unmatched_candidate_trks ]
		if self.debug_id in unmatched_trks_ids:
			print("candidates_association_and_KalmanUpdate : kill ")
		self.CandidatesTrackers = [candidate_trks for i,candidate_trks in enumerate(self.CandidatesTrackers) if i not in unmatched_candidate_trks]



		#unmatched_dets_ind2 are indices in the sublist  'unmatched_dets'
		#the return value should be in respect to  larger list: 'unmatched_dets_ind_input'
		unmatched_dets_ind_out = [unmatched_dets_ind_input[ind2] for ind2 in unmatched_dets_ind2 ]


		#else:
		#	unmatched_dets_ind_out=unmatched_dets

		return unmatched_dets_ind_out



	def active_association_and_KalmanUpdate(self, dets, info, active_trks, frame):

		"""
		#    1.2 associate & kalman update
		#    1.2.1 matching
		trk_innovation_matrix = None
		if self.metric == 'm_dis':
			trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in active_trks]
		matched_to_active_trks, unmatched_dets, unmatched_active_trks, cost, affi = \
			data_association(dets, active_trks, self.metric, self.metric_thres_1, self.algm, trk_innovation_matrix,points=points)
		# print_log('detections are', log=self.log, display=False)
		# print_log(dets, log=self.log, display=False)
		# print_log('tracklets are', log=self.log, display=False)
		# print_log(trks, log=self.log, display=False)
		# print_log('matched indexes are', log=self.log, display=False)
		# print_log(matched, log=self.log, display=False)
		# print_log('raw affinity matrix is', log=self.log, display=False)
		# print_log(affi, log=self.log, display=False)
		#    1.2.2 kalman update
		# update trks with matched detection measurement
		self.ActiveTrackers=self.KalmanUpdate(matched_to_active_trks, unmatched_active_trks,
											  dets, info, self.ActiveTrackers, frame,
											  bool_update_active=True)
		"""

		high_ind, low_ind = self.splitDetection(info,score_threshold=self.score_thres)

		# first association - high detection to activeTrackers
		high_unmatched_dets = [dets[i] for i in high_ind]
		matched_to_active_trks_high, unmatched_dets_ind2_high, unmatched_active_trks_high, cost, affi = \
			data_association(high_unmatched_dets, active_trks, self.metric, self.metric_thres_1, self.algm, None, points=None)

		# second association - low detection the remaining active
		remain_track_active = [active_trks[i] for i in unmatched_active_trks_high]
		low_unmatched_dets = [dets[i] for i in low_ind]

		matched_to_active_trks_low, unmatched_dets_ind2_low, unmatched_active_trks_low, cost2, affi_2 = \
			data_association(low_unmatched_dets, remain_track_active, self.metric,self.metric_thres_2, self.algm, None,
							 points=None)

		###################################
		###  matched_to_active_trks   ### :
		###################################
		# matched_to_active_trks_high detection part are indices in the sublist  'high_unmatched_dets'
		# the return value should be in respect to  larger list: 'dets'
		matched_to_active_trks_high_out = np.array([ np.stack((high_ind[m[0]], m[1]), axis=0) for m in matched_to_active_trks_high ])

		# 'matched_to_active_trks_low' detection & tracking part are indices in the sublist  'low_unmatched_dets' and 'remain_track_candidates'
		# the return value should be in respect to  larger list: 'dets' and 'active_trks'
		matched_to_active_trks_low_out =  np.array([ np.stack((low_ind[m[0]], unmatched_active_trks_high[m[1]]), axis=0)
											   for m in matched_to_active_trks_low ])
		if len(matched_to_active_trks_high_out)!=0 and len(matched_to_active_trks_low_out)!=0:
			matched_to_active_trks = np.vstack((matched_to_active_trks_high_out,matched_to_active_trks_low_out))
		elif  len(matched_to_active_trks_high_out)==0 and len(matched_to_active_trks_low_out)!=0:
			matched_to_active_trks = matched_to_active_trks_low_out
		elif len(matched_to_active_trks_high_out)!=0 and len(matched_to_active_trks_low_out)==0:
			matched_to_active_trks = matched_to_active_trks_high_out
		else:
			matched_to_active_trks =np.array([])

		###########################################
		###  unmatch matched_to_active_trks   ### :
		###########################################

		# 'unmatched_active_trks_low'  tracking part are indices in the sublist  'remain_track_active'
		# the return value should be in respect to  larger list: 'active_trks'
		unmatched_active_trks =[ unmatched_active_trks_high[i] for i in unmatched_active_trks_low ]

		# kalman update
		self.ActiveTrackers= self.KalmanUpdate(matched_to_active_trks, unmatched_active_trks,
											   dets, info,
											   self.ActiveTrackers, frame, bool_update_active=False)


		unmatched_trks_ids = [active_trks.id for i, active_trks in enumerate(self.ActiveTrackers) if
							  i in unmatched_active_trks]
		#if self.debug_id in unmatched_trks_ids:
		#	print("active_association_and_KalmanUpdate : kill ")

		# unmatched_trks are false alarm trackers, removed from the candidate list
		#self.ActiveTrackers = [candidate_trks for i, candidate_trks in enumerate(self.ActiveTrackers) if
		#						   i not in unmatched_active_trks]

		# unmatched_dets_ind2_high are indices in the sublist  'high_unmatched_dets'
		# the return value should be in respect to  larger list: 'dets'
		unmatched_dets_ind_high_out = [high_ind[i].astype(np.int64) for i in unmatched_dets_ind2_high]


		# unmatched_dets_ind2_low are indices in the sublist  'low_unmatched_dets'
		# the return value should be in respect to  larger list: 'dets'
		unmatched_dets_ind_low_out = [low_ind[i].astype(np.int64) for i in unmatched_dets_ind2_low]


		# there is an error if they share common indexes
		assert(len(np.intersect1d(unmatched_dets_ind_high_out,unmatched_dets_ind_low_out))==0)

		return unmatched_dets_ind_high_out,unmatched_dets_ind_low_out


	def candidates_association_and_KalmanUpdate_2(self, dets, unmatched_dets_ind_input, info, candidates_trks, frame):

		unmatched_dets = [d for ind, d in enumerate(dets) if ind in unmatched_dets_ind_input] # boxes
		unmatched_info = [i for ind, i in enumerate(info) if ind in unmatched_dets_ind_input] # info
		unmatched_info = np.array(unmatched_info)


		high_ind, low_ind = self.splitDetection(unmatched_info,score_threshold=self.score_thres)

		# first association - high detection to candidate
		high_unmatched_dets = [unmatched_dets[i] for i in high_ind]
		matched_to_candidates_trks_high, unmatched_dets_ind2_high, unmatched_candidate_trks_high, cost, affi = \
			data_association(high_unmatched_dets, candidates_trks, self.metric, self.metric_thres_1, self.algm, None, points=None)

		# second association - low detection the remaining candidate
		remain_track_candidates = [candidates_trks[i] for i in unmatched_candidate_trks_high]
		low_unmatched_dets = [unmatched_dets[i] for i in low_ind]

		matched_to_candidates_trks_low, unmatched_dets_ind2_low, unmatched_candidate_trks_low, cost2, affi_2 = \
			data_association(low_unmatched_dets, remain_track_candidates, self.metric,self.metric_thres_2, self.algm, None,
							 points=None)



		###################################
		###  matched_to_candidates_trks ### :
		###################################
		# matched_to_candidates_trks_high detection part are indices in the sublist  'high_unmatched_dets'
		# the return value should be in respect to  larger list: 'unmatched_dets'
		# unmatched_dets_ind_out = [unmatched_dets_ind_input[ind2] for ind2 in unmatched_dets_ind2]
		matched_to_candidates_trks_high_out = np.array([ np.stack((high_ind[m[0]], m[1]), axis=0) for m in matched_to_candidates_trks_high ])

		# 'matched_to_candidates_trks_low' detection & tracking part are indices in the sublist  'low_unmatched_dets' and 'remain_track_candidates'
		# the return value should be in respect to  larger list: 'unmatched_dets' and 'candidates_trks'
		matched_to_candidates_trks_low_out =  np.array([ np.stack((low_ind[m[0]], unmatched_candidate_trks_high[m[1]]), axis=0)
											   for m in matched_to_candidates_trks_low ])
		if len(matched_to_candidates_trks_high_out)!=0 and len(matched_to_candidates_trks_low_out)!=0:
			matched_to_candidates_trks = np.vstack((matched_to_candidates_trks_high_out,matched_to_candidates_trks_low_out))
		elif  len(matched_to_candidates_trks_high_out)==0 and len(matched_to_candidates_trks_low_out)!=0:
			matched_to_candidates_trks = matched_to_candidates_trks_low_out
		elif len(matched_to_candidates_trks_high_out)!=0 and len(matched_to_candidates_trks_low_out)==0:
			matched_to_candidates_trks = matched_to_candidates_trks_high_out
		else:
			matched_to_candidates_trks =np.array([])

		###########################################
		###  unmatch matched_to_candidates_trks ### :
		###########################################

		# 'unmatched_candidate_trks_low'  tracking part are indices in the sublist  'remain_track_candidates'
		# the return value should be in respect to  larger list: 'candidates_trks'
		unmatched_candidate_trks =[ unmatched_candidate_trks_high[i] for i in unmatched_candidate_trks_low ]

		# kalman update
		self.CandidatesTrackers = self.KalmanUpdate(matched_to_candidates_trks, unmatched_candidate_trks,
													unmatched_dets, unmatched_info,
													self.CandidatesTrackers, frame, bool_update_active=False)


		unmatched_trks_ids = [candidate_trks.id for i, candidate_trks in enumerate(self.CandidatesTrackers) if
							  i in unmatched_candidate_trks]
		if self.debug_id in unmatched_trks_ids:
			print("candidates_association_and_KalmanUpdate : kill ")

		# unmatched_trks are false alarm trackers, removed from the candidate list
		self.CandidatesTrackers = [candidate_trks for i, candidate_trks in enumerate(self.CandidatesTrackers) if
								   i not in unmatched_candidate_trks]

		# unmatched_dets_ind2_high are indices in the sublist  'unmatched_dets'
		# the return value should be in respect to  larger list: 'unmatched_dets_ind_input'
		unmatched_dets_ind_high_out = [unmatched_dets_ind_input[high_ind[i]].astype(np.int64) for i in unmatched_dets_ind2_high]

		# unmatched_dets_ind2_low are indices in the sublist  'unmatched_dets'
		# the return value should be in respect to  larger list: 'unmatched_dets_ind_input' and 'candidates_trks'
		unmatched_dets_ind_low_out = [unmatched_dets_ind_input[low_ind[i]].astype(np.int64) for i in unmatched_dets_ind2_low]

		# there is an error if they share common indexes
		assert(len(np.intersect1d(unmatched_dets_ind_high_out,unmatched_dets_ind_low_out))==0)

		return unmatched_dets_ind_high_out,unmatched_dets_ind_low_out

	def trks_status_update(self,frameNum):
		# move candidate to active
		# move active to candidate
		# kill candidates

		# active to candidate
		pop_idx = len(self.ActiveTrackers)
		for at, active_trk in enumerate(reversed(self.ActiveTrackers)):
			pop_idx -=1
			if active_trk.time_since_update >= self.max_age: # living on predict alone,
				if active_trk.id == self.debug_id:
					print(" trks_status_update: from active trk to candidate trk.")
				self.CandidatesTrackers.append(active_trk) # move to CandidatesTrackers
				self.ActiveTrackers.pop(pop_idx)                # remove from ActiveTrackers

		# candidate to active
		# candidate kill
		pop_idx=len(self.CandidatesTrackers)
		for ct, candidate_trk in enumerate(reversed(self.CandidatesTrackers)):
			pop_idx-=1
			if (candidate_trk.time_since_update < self.max_age) and (candidate_trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
				if candidate_trk.id == self.debug_id:
					print(" trks_status_update: from candidate trk to active trk.")
				self.ActiveTrackers.append(candidate_trk)      # move to ActiveTrackers
				self.CandidatesTrackers.pop(pop_idx)           # remove from CandidatesTrackers
			elif (candidate_trk.time_since_update >= self.death_age):
				if candidate_trk.id == self.debug_id:
					print(" trks_status_update: killed id", self.debug_id)
				self.CandidatesTrackers.pop(pop_idx)           # remove from CandidatesTrackers

	def output(self,frameNum):
		# output exiting tracks that have been stably associated, i.e., >= min_hits
		# and also delete tracks that have appeared for a long time, i.e., >= max_age

		num_trks = len(self.ActiveTrackers)
		results = []
		for trk in reversed(self.ActiveTrackers):
			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
			d = Box3D.array2bbox(trk.kf.x[:7].reshape((7, )))     # bbox location self
			d = Box3D.bbox2array_raw(d)

			#if ((trk.time_since_update < self.max_age)
			#		and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
			results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1))
			#num_trks -= 1

			if trk.id == self.debug_id:
				print('output :')
				print(trk.kf.x.reshape((-1)))
				print('\n current velocity')
				print(trk.get_velocity())

				self.saveStateX(frameNum, trk.kf.x, file_name=self.debug_kalman_output_file)

			# deadth, remove dead tracklet
			#if (trk.time_since_update >= self.death_age):
			#	self.trackers.pop(num_trks)

		return results

	def output_candidate(self,frameNum):
		# output exiting tracks that have been stably associated, i.e., >= min_hits
		# and also delete tracks that have appeared for a long time, i.e., >= max_age

		num_trks = len(self.CandidatesTrackers)
		results = []
		for trk in reversed(self.CandidatesTrackers):
			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
			d = Box3D.array2bbox(trk.kf.x[:7].reshape((7, )))     # bbox location self
			d = Box3D.bbox2array_raw(d)

			#if ((trk.time_since_update < self.max_age)
			#		and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
			results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1))
			#num_trks -= 1

			if trk.id == self.debug_id:
				print('output :')
				print(trk.kf.x.reshape((-1)))
				print('\n current velocity')
				print(trk.get_velocity())

				self.saveStateX(frameNum, trk.kf.x, file_name=self.debug_kalman_output_file)

			# deadth, remove dead tracklet
			#if (trk.time_since_update >= self.death_age):
			#	self.trackers.pop(num_trks)

		return results

	def track(self, dets_all, frame, seq_name,points=None):
		"""
		Params:
		  	dets_all: dict
				dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
				info: a array of other info for each det
			frame:    str, frame number, used to query ego pose
		Requires: this method must be called once for each frame even with empty detections.
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""
		dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
		#if self.debug_id: print('\nframe is %s' % frame)

		# logging
		print_str = '\n\n*****************************************\n\nprocessing seq_name/frame %s/%d' % (seq_name, frame)
		print_log(print_str, log=self.log, display=False)
		self.frame_count += 1

		## recall the last frames of outputs for computing ID correspondences during affinity processing
		##self.id_past_output = copy.copy(self.id_now_output)
		##self.id_past = [trk.id for trk in self.trackers]

		# process detection format
		dets = self.process_dets(dets) # converts structures:  dets (np.array) to dets (Box3D)

		# 1. stage 1 : track of active trackers
		#    1.1 kalman predict
		# tracks propagation based on velocity
		active_trks = self.KalmanPrediction(self.ActiveTrackers, frame,bool_predict_active=True)

		#    1.2 associate & kalman update
		unmatched_dets_ind_high_out,unmatched_dets_ind_low_out = self.active_association_and_KalmanUpdate(dets, info, active_trks, frame)
		unmatched_dets_ind =  np.union1d(unmatched_dets_ind_high_out,unmatched_dets_ind_low_out)

		#    1.3 detection refine: remove  unmatched detection that are too close to active tracks,
		#print("refine_detection frame : " +str(frame))
		#unmatched_dets_ind = refine_detection(dets, unmatched_dets_ind, active_trks, self.refine_thresh)
		#unmatched_dets_ind = unmatched_dets

		# 2. stage 2 : track of candidate trackers
		#    2.1 kalman predict
		candidates_trks = self.KalmanPrediction(trks=self.CandidatesTrackers,frameNum=frame,bool_predict_active=False)
		#    2.1 associate & kalman update
		unmatched_dets_ind_high_out, unmatched_dets_ind_low_out = self.candidates_association_and_KalmanUpdate_2(dets,unmatched_dets_ind, info,candidates_trks,																																  frame)
		unmatched_dets_out = unmatched_dets_ind_high_out

		# 3. stage 3 : create and initialize new trackers for unmatched detections
		#    3.1 create and initialize new trackers for unmatched detections
		new_id_list = self.tracklet_generation(dets, info, unmatched_dets_out,frame)

		# 4. stage 4 :  target linking

		self.trks_status_update(frame)

		# output existing valid tracks
		results = self.output(frame)
		#results = self.output_candidate(frame)
		if len(results) > 0: results = [np.concatenate(results)]		# h,w,l,x,y,z,theta, ID, other info, confidence
		else:            	 results = [np.empty((0, 15))]
		self.id_now_output = results[0][:, 7].tolist()					# only the active tracks that are outputed

		# post-processing affinity to convert to the affinity between resulting tracklets
		#if self.affi_process:
		#	affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)
		#  # print_log('processed affinity matrix is', log=self.log, display=False)
		#  # print_log(affi, log=self.log, display=False)

		# logging
		#print_log('\ntop-1 cost selected', log=self.log, display=False)
		#print_log(cost, log=self.log, display=False)
		#for result_index in range(len(results)):
		#	print_log(results[result_index][:, :8], log=self.log, display=False)
		#	print_log('', log=self.log, display=False)

		return results#, affi

	def splitDetection(self,info,score_threshold):
		high_ind, low_ind =[], []
		if len(info)!=0:
			high_ind = np.where(info[: ,6] > score_threshold)[0]
			low_ind  = np.array([ i for i in range(len(info)) if i not in high_ind])
		return high_ind,low_ind





