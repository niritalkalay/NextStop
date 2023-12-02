from scipy.optimize import linear_sum_assignment
import numpy as np
from abc import ABC, abstractmethod
import torch
import pandas as pd
from collections import defaultdict
import cProfile as profile # https://stackoverflow.com/questions/32926847/profiling-a-python-program-with-pycharm-or-any-other-ide\
import time
class _BaseMetric(ABC):
    @abstractmethod
    def __init__(self):
        self.plottable = False
        self.integer_fields = []
        self.float_fields = []
        self.array_labels = []
        self.integer_array_fields = []
        self.float_array_fields = []
        self.fields = []
        self.summary_fields = []
        self.registered = False

    #####################################################################
    # Abstract functions for subclasses to implement

    @abstractmethod
    def eval_sequence(self, data):
        ...

    @abstractmethod
    def combine_sequences(self, all_res):
        ...

    @abstractmethod
    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        ...

    @ abstractmethod
    def combine_classes_det_averaged(self, all_res):
        ...

    def plot_single_tracker_results(self, all_res, tracker, output_folder, cls):
        """Plot results of metrics, only valid for metrics with self.plottable"""
        if self.plottable:
            raise NotImplementedError('plot_results is not implemented for metric %s' % self.get_name())
        else:
            pass

    #####################################################################
    # Helper functions which are useful for all metrics:

    @classmethod
    def get_name(cls):
        return cls.__name__

    @staticmethod
    def _combine_sum(all_res, field):
        """Combine sequence results via sum"""
        return sum([all_res[k][field] for k in all_res.keys()])

    @staticmethod
    def _combine_weighted_av(all_res, field, comb_res, weight_field):
        """Combine sequence results via weighted average"""
        return sum([all_res[k][field] * all_res[k][weight_field] for k in all_res.keys()]) / np.maximum(1.0, comb_res[
            weight_field])

    def print_table(self, table_res, tracker, cls):
        """Prints table of results for all sequences"""
        print('')
        metric_name = self.get_name()
        self._row_print([metric_name + ': ' + tracker + '-' + cls] + self.summary_fields)
        for seq, results in sorted(table_res.items()):
            if seq == 'COMBINED_SEQ':
                continue
            summary_res = self._summary_row(results)
            self._row_print([seq] + summary_res)
        summary_res = self._summary_row(table_res['COMBINED_SEQ'])
        self._row_print(['COMBINED'] + summary_res)

    def _summary_row(self, results_):
        vals = []
        for h in self.summary_fields:
            if h in self.float_array_fields:
                vals.append("{0:1.5g}".format(100 * np.mean(results_[h])))
            elif h in self.float_fields:
                vals.append("{0:1.5g}".format(100 * float(results_[h])))
            elif h in self.integer_fields:
                vals.append("{0:d}".format(int(results_[h])))
            else:
                raise NotImplementedError("Summary function not implemented for this field type.")
        return vals

    @staticmethod
    def _row_print(*argv):
        """Prints results in an evenly spaced rows, with more space in first row"""
        if len(argv) == 1:
            argv = argv[0]
        to_print = '%-35s' % argv[0]
        for v in argv[1:]:
            to_print += '%-10s' % str(v)
        print(to_print)

    def summary_results(self, table_res):
        """Returns a simple summary of final results for a tracker"""
        return dict(zip(self.summary_fields, self._summary_row(table_res['COMBINED_SEQ'])))

    def detailed_results(self, table_res):
        """Returns detailed final results for a tracker"""
        # Get detailed field information
        detailed_fields = self.float_fields + self.integer_fields
        for h in self.float_array_fields + self.integer_array_fields:
            for alpha in [int(100*x) for x in self.array_labels]:
                detailed_fields.append(h + '___' + str(alpha))
            detailed_fields.append(h + '___AUC')

        # Get detailed results
        detailed_results = {}
        for seq, res in table_res.items():
            detailed_row = self._detailed_row(res)
            if len(detailed_row) != len(detailed_fields):
                raise ValueError(
                    'Field names and data have different sizes (%i and %i)' % (len(detailed_row), len(detailed_fields)))
            detailed_results[seq] = dict(zip(detailed_fields, detailed_row))
        return detailed_results

    def _detailed_row(self, res):
        detailed_row = []
        for h in self.float_fields + self.integer_fields:
            detailed_row.append(res[h])
        for h in self.float_array_fields + self.integer_array_fields:
            for i, alpha in enumerate([int(100 * x) for x in self.array_labels]):
                detailed_row.append(res[h][i])
            detailed_row.append(np.mean(res[h]))
        return detailed_row


class Identity(_BaseMetric):
    """Class which implements the ID metrics"""
    # https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/_base_metric.py

    def __init__(self, n_classes, device = None, ignore = None, min_points = 30,THRESHOLD=0.5):

        super().__init__()
        self.integer_fields = ['IDTP', 'IDFN', 'IDFP','Dets' ,'GT_Dets','IDs', 'GT_IDs']
        self.float_fields = ['IDF1', 'IDR', 'IDP']
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = self.fields

        self.n_classes = n_classes
        assert (device == None)
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array([n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
        self.num_stuff = 8

        # Configuration options:
        self.threshold = float(THRESHOLD)

        self.min_points = min_points
        self.count = 0

        self.unique_gt_ids = [[] for n in range(self.n_classes) if n not in self.ignore]
        self.unique_tracker_ids = [[] for n in range(self.n_classes) if n not in self.ignore]
        self.num_tracker_dets= [0 for n in range(self.n_classes) if n not in self.ignore]
        self.num_gt_dets =[0 for n in range(self.n_classes) if n not in self.ignore]

        self.data =[]
        for n in range(self.n_classes):
            if n not in self.ignore:
                self.data.append({
               'num_gt_ids': int(),
               'num_tracker_ids': int(),
               'num_gt_dets': int(),
               'num_tracker_dets': int(),
               'gt_ids': list(),
               #'gt_dets': list(),
               'tracker_ids': list(),
               #'tracker_dets':list(),
               'similarity_scores': list(),
                })
        self.data = np.array(self.data)


    def _check_unique_ids(self, after_preproc=False):
        """Check the requirement that the tracker_ids and gt_ids are unique per timestep"""
        # taken from https://github.com/JonathonLuiten/TrackEval
        for i in range(len(self.data)):
            gt_ids = self.data[i]['gt_ids']
            tracker_ids = self.data[i]['tracker_ids']
            for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):
                if len(tracker_ids_t) > 0:
                    unique_ids, counts = np.unique(tracker_ids_t, return_counts=True)
                    if np.max(counts) != 1:
                        duplicate_ids = unique_ids[counts > 1]
                        exc_str_init = 'Tracker predicts the same ID more than once in a single timestep ' \
                                       '(seq: %s, frame: %i, ids:' % (-1, t + 1)
                        exc_str = ' '.join([exc_str_init] + [str(d) for d in duplicate_ids]) + ')'
                        if after_preproc:
                            exc_str_init += '\n Note that this error occurred after preprocessing (but not before), ' \
                                            'so ids may not be as in file, and something seems wrong with preproc.'
                        raise ValueError(exc_str)
                if len(gt_ids_t) > 0:
                    unique_ids, counts = np.unique(gt_ids_t, return_counts=True)
                    if np.max(counts) != 1:
                        duplicate_ids = unique_ids[counts > 1]
                        exc_str_init = 'Ground-truth has the same ID more than once in a single timestep ' \
                                       '(seq: %s, frame: %i, ids:' % (-1, t + 1)
                        exc_str = ' '.join([exc_str_init] + [str(d) for d in duplicate_ids]) + ')'
                        if after_preproc:
                            exc_str_init += '\n Note that this error occurred after preprocessing (but not before), ' \
                                            'so ids may not be as in file, and something seems wrong with preproc.'
                        raise ValueError(exc_str)

    def eval_sequence(self):

        Nclass = len(self.data)

        """Calculates ID metrics for one sequence"""
        # Initialise results
        res = np.array([{} for i in range(Nclass)])
        for i in range(1,Nclass):
            for field in self.fields:
                res[i][field] = 0

            res[i]['Dets'] = self.data[i]['num_tracker_dets']
            res[i]['GT_Dets'] = self.data[i]['num_gt_dets']

            # Return result quickly if tracker or gt sequence is empty
            if self.data[i]['num_tracker_dets'] == 0:
                res[i]['IDFN'] = self.data[i]['num_gt_dets']
                continue
            if self.data[i]['num_gt_dets'] == 0:
                res[i]['IDFP'] = self.data[i]['num_tracker_dets']
                continue

            # Variables counting global association
            potential_matches_count = np.zeros((self.data[i]['num_gt_ids'], self.data[i]['num_tracker_ids']))
            gt_id_count = np.zeros(self.data[i]['num_gt_ids'])
            gt_ids = np.unique(np.concatenate(self.data[i]['gt_ids'], axis=0))
            tracker_id_count = np.zeros(self.data[i]['num_tracker_ids'])
            tracker_ids  = np.unique(np.concatenate(self.data[i]['tracker_ids'], axis=0))

            # First loop through each timestep and accumulate global track information.
            for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(self.data[i]['gt_ids'], self.data[i]['tracker_ids'])):
                # Count the potential matches between ids in each timestep
                matches_mask = np.greater_equal(self.data[i]['similarity_scores'][t], self.threshold)
                match_idx_gt, match_idx_tracker = torch.nonzero(matches_mask,as_tuple=True)


                gt_ind =[]
                for match_gt_el in match_idx_gt:
                    gt_ind.append(np.where(gt_ids==gt_ids_t[match_gt_el].detach().cpu().numpy())[0])
                pred_ind = []
                for match_pred_el in match_idx_tracker:
                    pred_ind.append(np.where(tracker_ids == tracker_ids_t[match_pred_el].detach().cpu().numpy())[0])

                potential_matches_count[gt_ind, pred_ind] += 1

                gt_ind = []
                for gt_t in gt_ids_t.detach().cpu().numpy():
                    gt_ind.append(np.where(gt_ids == gt_t)[0])

                pred_ind = []
                for tracker_t in tracker_ids_t.detach().cpu().numpy():
                    pred_ind.append(np.where(tracker_ids == tracker_t)[0])

                # Calculate the total number of dets for each gt_id and tracker_id.
                if len(gt_ind)!=0:
                    gt_id_count[np.array(gt_ind)] += 1
                if len(pred_ind)!=0:
                    tracker_id_count[np.array(pred_ind)] += 1

            # Calculate optimal assignment cost matrix for ID metrics
            num_gt_ids = self.data[i]['num_gt_ids']
            num_tracker_ids = self.data[i]['num_tracker_ids']
            fp_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
            fn_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
            fp_mat[num_gt_ids:, :num_tracker_ids] = 1e10
            fn_mat[:num_gt_ids, num_tracker_ids:] = 1e10
            for gt_id in range(num_gt_ids):
                fn_mat[gt_id, :num_tracker_ids] = gt_id_count[gt_id]
                fn_mat[gt_id, num_tracker_ids + gt_id] = gt_id_count[gt_id]
            for tracker_id in range(num_tracker_ids):
                fp_mat[:num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
                fp_mat[tracker_id + num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
            fn_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count
            fp_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count

            # Hungarian algorithm
            match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)

            # Accumulate basic statistics
            res[i]['IDFN'] = fn_mat[match_rows, match_cols].sum().astype(np.int)
            res[i]['IDFP'] = fp_mat[match_rows, match_cols].sum().astype(np.int)
            res[i]['IDTP'] = (gt_id_count.sum() - res[i]['IDFN']).astype(np.int)


            #res[i]['Dets'] = self.data[i]['num_tracker_dets']
            #res[i]['GT_Dets'] = self.data[i]['num_gt_dets']
            #res[i]['IDs'] = self.data[i]['num_tracker_ids']
            #res[i]['GT_IDs'] = self.data[i]['num_gt_ids']





            # Calculate final ID scores
            res[i] = self._compute_final_fields(res[i])
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        things_dict = dict(enumerate(all_res[1:9].flatten(), 1))
        for field in self.integer_fields:
            if ignore_empty_classes:
                res[field] = self._combine_sum({k: v for k, v in things_dict.items()
                                                if v['IDTP'] + v['IDFN'] + v['IDFP'] > 0 + np.finfo('float').eps},
                                               field)
            else:

                res[field] = self._combine_sum({k: v for k, v in things_dict.items()}, field)
        for field in self.float_fields:
            if ignore_empty_classes:
                res[field] = np.mean([v[field] for v in things_dict.values()
                                      if v['IDTP'] + v['IDFN'] + v['IDFP'] > 0 + np.finfo('float').eps], axis=0)
            else:
                res[field] = np.mean([v[field] for v in things_dict.values()], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        things_dict = dict(enumerate(all_res[1:9].flatten(), 1))
        for field in self.integer_fields:
            res[field] = self._combine_sum(things_dict, field)
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res['IDR'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFN'])
        res['IDP'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFP'])
        res['IDF1'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + 0.5 * res['IDFP'] + 0.5 * res['IDFN'])
        return res

    def addBatch(self,x_sem_row, x_inst_row, y_sem_row, y_inst_row):
        self.addBatchID(x_sem_row, x_inst_row, y_sem_row, y_inst_row)

    def addBatchID(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
        # In outer section of code
        #pr = profile.Profile()

        #pr.enable()
        self._check_unique_ids()

        y_sem_row = torch.from_numpy(y_sem_row).long()
        y_inst_row = torch.from_numpy(y_inst_row.astype(np.int32)).long()

        x_sem_row = torch.from_numpy(x_sem_row).long()
        x_inst_row = torch.from_numpy(x_inst_row.astype(np.int32)).long()

        unique_cat_gt = np.unique(y_sem_row)
        for cls_id in unique_cat_gt:

            if cls_id ==0 or cls_id>8:
                continue


            # init
            self.data[cls_id]['gt_ids'].append([])
            self.data[cls_id]['tracker_ids'].append([])
            self.data[cls_id]['similarity_scores'].append([])


            m_gt  = y_sem_row == cls_id
            m_prd = x_sem_row == cls_id

            gt_ids =torch.unique(y_inst_row[m_gt])
            #gt_ids[np.where(gt_ids==0)[0]]=[]
            tracker_ids = torch.unique(x_inst_row[m_prd])
            #tracker_ids[np.where(tracker_ids == 0)[0]] = []


            # Calculate similarities for each timestep.
            #s= time.time()
            # todo: insert here filter by self.min_points_size
            similarity_scores = torch.zeros(len(gt_ids),len(tracker_ids), dtype=torch.double)
            for g,gt_l in enumerate(gt_ids):
                gt_label_points = np.where(y_inst_row == gt_l)[0]
                #m_gt_label_points = y_inst_row == gt_l
                for p, pred_l in enumerate(tracker_ids):
                    pred_label_points = np.where(x_inst_row == pred_l)[0]
                    #m_pred_label_points = x_inst_row == pred_l

                    #s = time.time()
                    overlap1 = len(np.intersect1d(gt_label_points,pred_label_points))
                    union1 = len(np.union1d(gt_label_points,pred_label_points))
                    similarity_scores[g, p] = overlap1 / union1
                    #print("my iou time ", time.time() - s)

                    #s = time.time()
                    #intersection2 = (m_pred_label_points[m_gt_label_points]).long().sum()#.data.cpu()[0]  # Cast to long to prevent overflows
                    #union2 = m_pred_label_points.long().sum() + m_gt_label_points.long().sum() - intersection2#.data.cpu()[0] - intersection
                    #print("torch iou time ", time.time() - s)
                    #if union == 0:
                    #    ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
                    #else:
                    #    ious.append(float(intersection) / float(max(union, 1)))



                    #s = time.time()
                    #overlap_m = m_pred_label_points * m_gt_label_points  # Logical AND
                    #union_m = (m_pred_label_points + m_gt_label_points) > 0  # Logical OR
                    #similarity_scores[g, p] = overlap_m.sum() / float(union_m.sum())
                    #print("mask iou time ", time.time() - s)

            #print("  Calculate similarities time ", time.time() - s)

            # Match tracker and gt dets (with hungarian algorithm)
            #s = time.time()
            unmatched_indices = np.arange(tracker_ids.shape[0])
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.clone()
                matching_scores[matching_scores < self.threshold - np.finfo('float').eps] = -10000
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = (matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps).detach().cpu().numpy()
                match_cols = match_cols[actually_matched_mask]
                unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)
            #print("  hungarian time ", time.time() - s)

            # Apply preprocessing to remove unwanted tracker dets.
            #s = time.time()
            to_remove_tracker = unmatched_indices
            self.data[cls_id]['tracker_ids'][-1] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Keep all ground truth detections
            self.data[cls_id]['gt_ids'][-1]=gt_ids
            self.data[cls_id]['similarity_scores'][-1]=similarity_scores
            #print("  preprocessing time ", time.time() - s)

            #s = time.time()
            self.unique_gt_ids[cls_id] += list(np.unique(self.data[cls_id]['gt_ids'][-1]))
            self.unique_gt_ids[cls_id] = list(np.unique(self.unique_gt_ids[cls_id]))
            self.unique_tracker_ids[cls_id] += list(np.unique(self.data[cls_id]['tracker_ids'][-1]))
            self.unique_tracker_ids[cls_id] = list(np.unique(self.unique_tracker_ids[cls_id]))
            self.num_tracker_dets[cls_id] = len(self.unique_tracker_ids[cls_id])
            self.num_gt_dets[cls_id] = len(self.unique_gt_ids[cls_id])

            # Record overview statistics.
            self.data[cls_id]['num_tracker_dets'] = self.num_tracker_dets[cls_id]
            self.data[cls_id]['num_gt_dets'] = self.num_gt_dets[cls_id]
            self.data[cls_id]['num_tracker_ids'] = len(self.unique_tracker_ids[cls_id])
            self.data[cls_id]['num_gt_ids'] = len(self.unique_gt_ids[cls_id])
            #print("  Record time ", time.time() - s)

        self._check_unique_ids(after_preproc=True)

        self.count += 1
        #pr.disable()
        #pr.dump_stats('profile.pstat') # to open: snakeviz profile.pstat

    #"""

    def print_results(self,res,combine_res ):


        data = []
        index = []
        for key, value in combine_res.items():
            IDF1 = "{:.2f}".format(value['IDF1'])
            IDR = "{:.2f}".format(value['IDR'])
            IDP = "{:.2f}".format(value['IDP'])
            IDTP = value['IDTP']
            IDFN = value['IDFN']
            IDFP = value['IDFP']

            Dets = value['Dets']
            GT_Dets = value['GT_Dets']
            # IDs = res[cls_id]['IDs']
            # GT_IDs = res[cls_id]['GT_IDs']

            data.append([IDF1, IDR, IDP, IDTP, IDFN, IDFP, Dets, GT_Dets])
            index.append(key)

        df = pd.DataFrame(data, columns=['IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP', 'Dets', 'GT_Dets'],
                          index=index)
        #df = df.style.set_caption('COMBINED')
        print(df)
        print("")

        ##########
        n_classes = len(res)
        data = []
        for cls_id in range(n_classes):
            if cls_id == 0 or cls_id > 8:
                continue


            IDF1 = "{:.2f}".format(res[cls_id]['IDF1'])
            IDR  = "{:.2f}".format(res[cls_id]['IDR'])
            IDP  = "{:.2f}".format(res[cls_id]['IDP'])
            IDTP = res[cls_id]['IDTP']
            IDFN = res[cls_id]['IDFN']
            IDFP = res[cls_id]['IDFP']

            Dets = res[cls_id]['Dets']
            GT_Dets = res[cls_id]['GT_Dets']
            #IDs = res[cls_id]['IDs']
            #GT_IDs = res[cls_id]['GT_IDs']




            data.append([IDF1,IDR,IDP,IDTP,IDFN,IDFP,Dets,GT_Dets])

        df2 = pd.DataFrame(data, columns=['IDF1', 'IDR', 'IDP', 'IDTP','IDFN', 'IDFP','Dets','GT_Dets'],
                          index=['car', 'bicycle','motorcycle', 'truck','other-vehicle','person','bicyclist','motorcyclist'])
        print(df2)


    def getID(self):

        res = self.eval_sequence()

        # combine classes

        combined_res = {}
        #res['COMBINED_SEQ'] = {}
        combined_res['cls_comb_cls_av'] = {}
        combined_res['cls_comb_det_av'] = {}

        combined_res['cls_comb_cls_av']= self.combine_classes_class_averaged(res)
        combined_res['cls_comb_det_av'] = self.combine_classes_det_averaged(res)
        #res['COMBINED_SEQ']c['ID'] = \
        #    self.combine_classes_class_averaged(res)
        #res['COMBINED_SEQ']['cls_comb_det_av']['ID'] = \
        #    self.combine_classes_det_averaged(res)


        self.print_results(res,combined_res)

        # results on all things class





        return self.data, res , combined_res





