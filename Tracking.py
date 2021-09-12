import cv2 as cv2
import numpy as np
import glob
import random
import time
import re
import csv
import optparse
from PIL import Image
from os import listdir
from os.path import isfile, join
import math
from scipy.optimize import linear_sum_assignment

## To run this tracker: python Tracker.py --frames [your_image_path] ###

class Uncertain:
    def __init__(self, coord, chances=5):
        self.coord = coord
        self.chances = chances


class State:
    def __init__(self):
        self.x, self.y= 0, 0
    def get_centroid(self):
        return [int(self.x), int(self.y)]

    def set_centroid(self, cx, cy):
        self.x, self.y = cx, cy

# Helper Class
class helper:
    @staticmethod
    def get_curr_time():
        return int(round(time.time() * 1000))

    @staticmethod 
    def get_ids(alst):
        return [label_id for (_, label_id) in alst]

    @staticmethod
    def remove_from_array(arr, target):
        for idx in range(len(arr)):
            if target == arr[idx]:
                return arr[:idx] + arr[idx+1:]
        return arr
    
    @staticmethod
    def distance(obj1,obj2):
        return np.linalg.norm(np.array(obj1)-np.array(obj2))        

    @staticmethod
    def rand_color(upper_range=1000):
        color_hash ={}
        for k in range(upper_range):
            color_hash[k]=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        return color_hash

    @staticmethod
    def paint(x_pos,color_hash,frame):
        hash= {}
        for frame_num in range(len(x_pos)):
            for detection in x_pos[frame_num]:
                if detection[1] in hash:
                    hash[detection[1]]['y'].append(detection[0][1])
                    hash[detection[1]]['x'].append(detection[0][0])
                else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                    hash[detection[1]]={'y':[detection[0][1]],'x':[detection[0][0]]}

        # iter = 0
        for key in hash:
            y= hash[key]['y']
            x= hash[key]['x']
            for i in range(len(x)):
                color = color_hash[key]
                frame = cv2.circle(frame, (x[i],y[i]), 3, color, -1)
        return frame


def preprocess_data(img_path):
    # get all files
    files = [join(img_path,f) for f in listdir(img_path) if isfile(join(img_path,f))]
    # init frame list
    img = []

    for i in range(len(files)):
        a_frame = cv2.imread(files[i])
        if a_frame is not None:
            new_shape = (a_frame.shape[1]//2, a_frame.shape[0]//2)
            a_frame = cv2.resize(a_frame, new_shape)
            img.append(a_frame)

    # compute average frame of cells
    avg_frame = np.average(np.array(img), axis=0).astype(np.uint8)
    
    # init location
    location = []
    for i in img:
        test_result = mainComputation(i, avg_frame)
        #print(test_result)
        location.append(test_result)

    return img, location    


    



# compute a frame(by minus average frame) to get all cells' centroid
def mainComputation(curr_frame, avg_frame):
    # the result is a list of[x,y,w,h]
    result = []
    
    frame_diff = cv2.absdiff(curr_frame, avg_frame)
    
    # Remove noise
    roi_mask = np.ones((frame_diff.shape[0], frame_diff.shape[1]))
    roi_mask = roi_mask.astype(np.int8)
    roi_mask[0:15, 190:] = 0
    frame_roi = cv2.bitwise_and(frame_diff, frame_diff, mask=roi_mask)
     # Thresholding
    # frame_th = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 5)
    _, frame_th = cv2.threshold(frame_roi, 10, 255, cv2.THRESH_BINARY)
    frame_blur = cv2.GaussianBlur(frame_th, (3, 3), 0)

            
    frame_morph = cv2.morphologyEx(frame_th, cv2.MORPH_CLOSE, (7, 7), iterations=1)
    frame_morph = cv2.morphologyEx(frame_morph, cv2.MORPH_OPEN, (7, 7), iterations=1)
    frame_morph = cv2.morphologyEx(frame_morph, cv2.MORPH_CLOSE, (11, 11), iterations=1)

    frame_blur = cv2.GaussianBlur(frame_morph, (9, 9), 0)
    _, frame_blur = cv2.threshold(frame_blur, 50, 255, cv2.THRESH_BINARY)

    gray_frame_blur = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    # Flood filling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_frame_blur, 4)
    n_obj = 0
    cur_frame_locations = []

    for stat, cent in zip(stats[1:], centroids[1:]):
        a_single_obj = []
        x, y, w, h = stat[:4]
        if stat[4] < 100:
            continue
        color = (0, 255, 0)
        n_obj += 1
        state = State()
        state.set_centroid(cent[0],cent[1])
        cur_frame_locations.append(state)
    return cur_frame_locations



class DataAssociation:
    @staticmethod
    def start_associating(x_pred, frame_measurements, v_pred, uncertain_hash):
        # find the nearest points with predict value
        uncertain_list = [[hd_item.coord, hd_id] for (hd_id,hd_item) in uncertain_hash.items()]
        all_remains_objs = x_pred + uncertain_list
        x_pred_locs_hash =  DataAssociation.matching(frame_measurements, all_remains_objs)
        # compute the matching results
        cur_frame_x_pred_labels, new_locs, left_loc_ids = DataAssociation.greedy_mapping(x_pred_locs_hash, uncertain_hash, frame_measurements)

        used_ids, not_used_ids_h = DataAssociation.calc_unused_ids(all_remains_objs, cur_frame_x_pred_labels, new_locs)
        
        # new bojects
        new_object_id = DataAssociation.get_new_object_id(uncertain_hash, x_pred)
        new_locs = DataAssociation.getting_new_object(new_locs, left_loc_ids, frame_measurements, new_object_id)

 
        # change hash
        used_ids, not_used_ids_h = DataAssociation.calc_unused_ids(x_pred, cur_frame_x_pred_labels, new_locs)
        uncertain_hash = DataAssociation.update_uncertain_hash(uncertain_hash, used_ids, not_used_ids_h)

        return cur_frame_x_pred_labels, new_locs, not_used_ids_h, uncertain_hash, used_ids

    @staticmethod
    def matching(frame_measurements, x_pred, gating_dist = 100):

        x_pred_locs_hash = {} 
        for j, meas in enumerate(frame_measurements):
            min_dist = float('inf')
            closest_x_pred_label = None

            # find nearest
            for _, x_pred_data in enumerate(x_pred):
                x_pred_coord, x_pred_label = x_pred_data

                meas_coord = [meas.get_centroid()[0], meas.get_centroid()[1]]

                dist = helper.distance(x_pred_coord, meas.get_centroid())     # distance from x_pred (prediction) to localization point
                if dist < min_dist:
                    min_dist = dist
                    closest_x_pred_label = x_pred_label

            if min_dist < gating_dist:
                if closest_x_pred_label in x_pred_locs_hash.keys():
                    x_pred_locs_hash[closest_x_pred_label].append((j, min_dist, meas.get_centroid()))
                else:
                    x_pred_locs_hash[closest_x_pred_label] = [(j, min_dist, meas.get_centroid())]

        return x_pred_locs_hash

    @staticmethod
    def greedy_mapping(x_pred_locs_hash, uncertain_hash, frame_measurements):
        cur_frame_x_pred_labels = []
        new_locs = []
        left_loc_ids = []
        seen_loc_ids = []
        for x_pred_key in x_pred_locs_hash.keys():
            loc_datas = x_pred_locs_hash[x_pred_key]
            if len(loc_datas) == 1:         
                loc_idx = loc_datas[0][0]
                loc_item = [frame_measurements[loc_idx].get_centroid(), x_pred_key]
                seen_loc_ids.append(loc_idx)
                if x_pred_key in uncertain_hash.keys():
                    new_locs.append(loc_item)
                else:
                    cur_frame_x_pred_labels.append(loc_item)
            else:
                loc_idxs = [loc_data[0] for loc_data in loc_datas]
                min_idx = None
                min_dist = float('inf')
                for loc_data in loc_datas:
                    loc_idx, loc_dist, _ = loc_data
                    if loc_dist < min_dist:
                        min_dist = loc_dist
                        min_idx = loc_idx

                loc_item = [frame_measurements[min_idx].get_centroid(), x_pred_key]
                seen_loc_ids.append(min_idx)

                if x_pred_key in uncertain_hash.keys():
                    new_locs.append(loc_item)
                else:
                    cur_frame_x_pred_labels.append(loc_item)     #update
                left_loc_ids += helper.remove_from_array(loc_idxs, min_idx) # left locs

                seen_loc_ids += left_loc_ids                

        not_seen_loc_ids = []
        for fm_idx in range(len(frame_measurements)):
            if fm_idx not in seen_loc_ids:
                not_seen_loc_ids.append(fm_idx)

        left_loc_ids += not_seen_loc_ids

        return cur_frame_x_pred_labels, new_locs, left_loc_ids
                    
    @staticmethod
    def get_new_object_id(uncertain_hash, x_pred):
        uncertain_ids = [hd_id for (hd_id, _) in uncertain_hash.items()]
        x_pred_ids = helper.get_ids(x_pred)
        obj_labels_np = np.array(x_pred_ids + uncertain_ids)
        new_obj_label = np.max(obj_labels_np) + 1
        return new_obj_label

    @staticmethod
    def update_uncertain_hash(uncertain_hash, used_ids, not_used_ids_h):
        # Update Uncertain Hash
        # Remove Uncertain that are used
        for used_id in used_ids:
            if used_id in uncertain_hash.keys():
                del uncertain_hash[used_id]

        # Add not_used into uncertain
        for (not_used_id, not_used_coord) in not_used_ids_h.items():
            uncertain_hash[not_used_id] = Uncertain(not_used_coord)

        # do updating tasks 
        uncertain_backup = list(uncertain_hash.keys()).copy()
        for hd_id in uncertain_backup:
            hd_item = uncertain_hash[hd_id]
            hd_item.chances -= 1
            if hd_item.chances <= 0:
                del uncertain_hash[hd_id]

        return uncertain_hash


    @staticmethod
    def getting_new_object(new_locs, left_loc_ids, frame_measurements, new_obj_label):
        for left_idx in range(len(left_loc_ids)):
            loc_idx = left_loc_ids[left_idx]
            loc_coord = frame_measurements[loc_idx].get_centroid()
            new_locs.append([loc_coord, new_obj_label])
            new_obj_label += 1

        return new_locs

    @staticmethod
    # Get all not used ids
    def calc_unused_ids(x_pred, cur_frame_x_pred_labels, new_locs):
        all_x_pred_ids = {obj_id: obj_coord for (obj_coord, obj_id) in x_pred}
        used_ids = helper.get_ids(cur_frame_x_pred_labels) + helper.get_ids(new_locs)
        
        for used_id in used_ids:
            if used_id in all_x_pred_ids.keys():
                del all_x_pred_ids[used_id]
        not_used_ids_h = all_x_pred_ids

        return used_ids, not_used_ids_h




class ABFilter:
    def __init__(self, location,img , data_association_fn, window_size = (600, 600)):
        self.location = location
        self.img = img
        self.data_association_fn = data_association_fn
        self.window_size = window_size
    # get the predict values    
    def get_x_pred(self,x_prev,v_prev):
        x_pred = []
        for (x_prev_coord, x_prev_id) in x_prev:
            found_id = False
            for(v_prev_coord, v_prev_id) in v_prev:
                if v_prev_id == x_prev_id:
                    found_id = True
                    x_pred_item = [[x_prev_coord[0] + v_prev_coord[0], x_prev_coord[1] + v_prev_coord[1]], v_prev_id]
                    x_pred.append(x_pred_item)
                    # break
            if found_id == False:
                x_pred_item = [x_prev_coord, x_prev_id]
                x_pred.append(x_pred_item)
                if SHOW_WARNING:
                    print("get_x_pred | id({}) not found".format(x_prev_id))

        return x_pred

    def update_velocity(self, v_preds, beta, residuals, new_locs):
        residual_hash = {}
        for _, residual in enumerate(residuals):
            residual_coord, residual_label = residual
            residual_hash[residual_label] = residual_coord
        # perform object wise math_logicion
        results = []
        # the below assumes element wise alignment, this may or may not hold
        for _, v_pred in enumerate(v_preds):
            v_pred_coord, v_pred_label = v_pred
            if v_pred_label in residual_hash.keys():
                residual = residual_hash[v_pred_label]
                est_x_coord, est_y_coord = v_pred_coord[0] + beta * residual[0], v_pred_coord[1] + beta * residual[1]
                est_coord = (est_x_coord, est_y_coord)
                results.append([est_coord, v_pred_label])
        for _, new_loc in enumerate(new_locs):
            loc_coord, loc_label = new_loc
            results.append([[0,0], loc_label])  

        return results


    def math_logic(self, cur_measurements, x_preds):
        res = []
        # calculating residuals
        for _, x_pred in enumerate(x_preds):
            x_pred_coord, x_pred_label = x_pred
            found_x_pred = False
            for _, cur_measurement in enumerate(cur_measurements):
                cm_coord, cm_label = cur_measurement
                if x_pred_label == cm_label:                    
                    x_residual = cm_coord[0] - x_pred_coord[0]
                    y_residual = cm_coord[1] - x_pred_coord[1]
                    residual = (x_residual, y_residual)
                    res.append([residual, cm_label])
                    found_x_pred = True
                    break
        return res

    def excute(self):
        while (True):
            cv2.namedWindow('trajectories',cv2.WINDOW_NORMAL)
            cv2.namedWindow('localization',cv2.WINDOW_NORMAL)
            # set parameters
            alpha = beta = 1
            x_prev, v_prev, x_pos_compiled, cur_measurements = [], [], [], []
            x_prev_actual = []
            color_hash = helper.rand_color()
            # Initialize x_prev, v_prev
            for i in range(len(self.location[0])): # first frame
                x,y = self.location[0][i].get_centroid()
                x_prev.append([[x,y],i])
                cur_measurements.append([[x,y],i])
                v_prev.append([[0,0],i])

            uncertain_hash = {}

            # Process frames of video
            for i,frame in enumerate(self.img):
                time0 = helper.get_curr_time()
                x_orig_prev = cur_measurements.copy()


                x_pred = self.get_x_pred(x_prev, v_prev)
                x_pred_copy = x_pred.copy()
                v_pred = v_prev
                # associating objects across frames
                cur_measurements, new_locs, not_used_ids_h, uncertain_hash, used_ids = self.data_association_fn(x_pred, self.location[i], v_pred, uncertain_hash)

                # update parameters
                res = self.math_logic(cur_measurements, x_pred)
                cur_measurements += new_locs
                x_est = self.pos_update(x_pred, alpha, res, new_locs)
                first_frame = (i == 0)
                v_est = self.update_velocity(v_pred, beta, res, new_locs) if not first_frame else v_pred

                v_prev = v_est
                x_prev = x_est           

                x_future = self.get_x_pred(x_prev, v_prev)
                x_pos_compiled.append(x_est)

                x_pos_curr_frame = []
                for x_pos_compiled_per_frame in x_pos_compiled:
                    keep_x_pos_compiled_per_frame = []
                    for (x_pos_coord, x_pos_id) in x_pos_compiled_per_frame:                    
                        if x_pos_id in used_ids:
                            keep_x_pos_compiled_per_frame.append([x_pos_coord, x_pos_id])
                    x_pos_curr_frame.append(keep_x_pos_compiled_per_frame)

                dimen = self.img[0].shape

                x_pos_compiled_frame = frame.copy()
                x_pos_compiled_frame = helper.paint(x_pos_curr_frame, color_hash, x_pos_compiled_frame)

                localization_frame = frame.copy()
                localization_frame = helper.paint([cur_measurements], color_hash, localization_frame)

                # Step 6: Set x_orig_prev
                x_orig_prev = cur_measurements

                if i >= 0: # and False:
                        cv2.imshow("trajectories",x_pos_compiled_frame)
                        cv2.resizeWindow('trajectories', self.window_size[0], self.window_size[1])

                        cv2.imshow("localization", localization_frame)
                        cv2.resizeWindow('localization', self.window_size[0], self.window_size[1])

                        if cv2.waitKey(1) & 0xFF == 27:   # continue
                            break
            cv2.destroyAllWindows()

    def pos_update(self, x_preds, alpha, residuals, new_locs):
        residual_hash = {}
        for _, residual in enumerate(residuals):
            residual_coord, residual_label = residual
            residual_hash[residual_label] = residual_coord
        # perform math_logicion
        results = []
        # do updating tasks
        for _, x_pred in enumerate(x_preds):
            x_pred_coord, x_pred_label = x_pred

            if x_pred_label in residual_hash.keys():
                residual = residual_hash[x_pred_label]
                est_x_coord, est_y_coord = x_pred_coord[0] + alpha * residual[0], x_pred_coord[1] + alpha * residual[1]
                est_coord = (est_x_coord, est_y_coord)
                results.append([est_coord, x_pred_label])
        for _, new_loc in enumerate(new_locs):
            loc_coord, loc_label = new_loc
            results.append([loc_coord, loc_label])

        return results

if __name__ == "__main__":

    print("starting processing...")
    parser = optparse.OptionParser()
    parser.add_option('--frames', dest='img_path', type='string', help='the path of frame files')

    (options, args) = parser.parse_args()
    img_src = options.img_path

    imgs, locations = preprocess_data(img_src)

    test_tracker = ABFilter(locations,imgs, data_association_fn = DataAssociation.start_associating, window_size=(600,600))
    test_tracker.excute()