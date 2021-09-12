import cv2
import numpy as np
import csv
from os import listdir
from os.path import isfile, join
import optparse





# compute a frame(by minus average frame) to get all cells' centroid
def mainComputation(curr_frame, avg_frame):
    # the result is a list of[x,y,w,h]
    result = []
    
    frame_diff = cv2.absdiff(curr_frame, avg_frame)
    
    # Remove boundary noise; Adding mask
    roi_mask = np.ones((frame_diff.shape[0], frame_diff.shape[1]))
    roi_mask = roi_mask.astype(np.int8)
    roi_mask[0:15, 190:] = 0
    frame_roi = cv2.bitwise_and(frame_diff, frame_diff, mask=roi_mask)
     # Thresholding
    # frame_th = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 5)
    _, frame_th = cv2.threshold(frame_roi, 10, 255, cv2.THRESH_BINARY)
    frame_blur = cv2.GaussianBlur(frame_th, (3, 3), 0)
    # _, frame_th = cv2.threshold(frame_blur, 50, 255, cv2.THRESH_BINARY)
            
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
        cur_frame_locations.append([int(cent[0]), int(cent[1])])

    # output the number of objects
    #print(n_obj)
    return cur_frame_locations




# prev_frame and curr_frame is 2d list which contains position of objects
def greedy_matching(prev_frame, curr_frame):
    prev_objNum = len(prev_frame)
    curr_objNum = len(curr_frame)
    matched_idx_p = []
    matched_idx_c = []
    isolated_idx_p = []
    isolated_idx_c = []
    if prev_objNum <= curr_objNum:
        # loop for checking matching pairs
        for idx_p in range(prev_objNum):
            an_obj = prev_frame[idx_p]
            # init the distance value
            minimum_distance = euclid_disance(an_obj, curr_frame[0])
            curr_match_idx_c = 0
            for idx_c in range(curr_objNum):
                new_distance = euclid_disance(an_obj, curr_frame[idx_c])
                if new_distance < minimum_distance and idx_c not in matched_idx_c:
                    # update the minimum distance and current match idx_c
                    minimum_distance = new_distance
                    curr_match_idx_c = idx_c
            matched_idx_p.append(idx_p)
            matched_idx_c.append(curr_match_idx_c)
        # loop for checking isolated idx_c
        if prev_objNum < curr_objNum:
            for idx_c in range(curr_objNum):
                if idx_c not in matched_idx_c:
                    isolated_idx_c.append(idx_c)    

    else:
        for idx_c in range(curr_objNum):
            an_obj = curr_frame[idx_c]
            # init the distance value
            minimum_distance = euclid_disance(an_obj, prev_frame[0])
            curr_match_idx_p = 0
            for idx_p in range(prev_objNum):
                new_distance = euclid_disance(an_obj, prev_frame[idx_p])
                if new_distance < minimum_distance and idx_p not in matched_idx_p:
                    # update the minimum distance and current match idx_c
                    minimum_distance = new_distance
                    curr_match_idx_p = idx_p
            matched_idx_p.append(curr_match_idx_p)
            matched_idx_c.append(idx_c)

        # loop for checking isolated idx_c
        if prev_objNum > curr_objNum:
            for idx_p in range(prev_objNum):
                if idx_p not in matched_idx_p:
                    isolated_idx_p.append(idx_p)       

    return matched_idx_p, matched_idx_c, isolated_idx_p, isolated_idx_c        


# obj1 and obj2 is the position
def euclid_disance(obj1, obj2):
    E_distance = (obj1[0] - obj2[0]) ** 2 + (obj1[1] - obj2[1]) ** 2

    return E_distance


if __name__ == "__main__":

    print("starting processing...")
    parser = optparse.OptionParser()
    parser.add_option('--frames', dest='img_path', type='string', help='the path of frame files')

    (options, args) = parser.parse_args()
    img_path = options.img_path

    files = [join(img_path,f) for f in listdir(img_path) if isfile(join(img_path,f))]

    img = []
    # img[23] is None!
    for i in range(len(files)):
        a_frame = cv2.imread(files[i])
        if a_frame is not None:
            new_shape = (a_frame.shape[1]//2, a_frame.shape[0]//2)
            a_frame = cv2.resize(a_frame, new_shape)
            img.append(a_frame)    
        

    # compute average frame
    avg_frame = np.average(np.array(img), axis=0).astype(np.uint8)  

    cells_location = []
    for i in img:
        test_result = mainComputation(i, avg_frame)
        #print(test_result)
        cells_location.append(test_result)      




    img_indx = 1
    # start num of bat objects
    start_len = len(cells_location[0])
    # init the start tagging list
    tag_list = []
    for idx in range(start_len):
        tag_list.append(idx)
    # for adding new tag
    tag_end = start_len    
        
    font = cv2.FONT_HERSHEY_SIMPLEX

    # while img_indx < len(files):
    while img_indx < len(files)-1:
        cur = img[img_indx]
        
        cur_frame_locations = cells_location[img_indx]
        cur_objNum = len(cur_frame_locations)
        prev_frame_locations = cells_location[img_indx-1]
        
        prev_match, curr_match, prev_isolated, curr_isolated = greedy_matching(prev_frame_locations, cur_frame_locations)
        

        #contour_output = cv2.cvtColor(np.zeros(np.shape(thres_output), dtype='uint8'), cv2.COLOR_GRAY2BGR)
        contour_output = img[img_indx][:,:,:].copy()
        
        # init the new tag list
        new_tag_list = []
        for tag in range(999):
            new_tag_list.append(0)
            
        print(len(prev_match))

        print(len(curr_match))
        for idx in range(len(prev_match)):
            tag_in_prev_frame = tag_list[prev_match[idx]]
            idx_in_cur_frame = curr_match[idx]
            # get the current obj pos
            obj_pos = cur_frame_locations[idx_in_cur_frame]
            # update tag list
            new_tag_list[idx_in_cur_frame] = tag_in_prev_frame 
            # draw a text there
            contour_output = cv2.putText(contour_output, str(tag_in_prev_frame), (obj_pos[0],obj_pos[1]), font, 1, (0,0,255))
            
        for new_obj_indx in curr_isolated:
            new_tag_list[new_obj_indx] = tag_end
            tag_end = tag_end + 1
        
        tag_list = new_tag_list    
        
        # Show in a window
        cv2.namedWindow("Contours", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Contours", contour_output)
        #cv2.imshow("Contours", cur)
        if cv2.waitKey(100)&0xFF == 27:
            break    
        
        img_indx += 1

    cv2.destroyAllWindows()