# ObjectTracking

## Problem Definition

<p>
Current problem is to track multiple objects in video, and our input is frame images of bats and cells: </p>
<p>We directly used the detection results and segmentation results of bats, and do detection and segmentation of cells on our own code</p>
<p>We made assumption that the nearest points between 2 sequential frames is very likely to be the same objects. So we apply GNNSF in data association
</p>
<p>To run the tracker: python Tracker.py --frames [your_image_path]</p>
<p>To run the test_greedy: python test_greedy.py --frames [your_image_path]</p>

## Method and Implementation

<p>1. Do detection and segmentation on bats and cells frames</p>
<p>To do this, we first calculate the average frame and use it to remove noise and background influence. Then we apply cv2.connectedComponentsWithStats to get 
features of segmented objects, among which we care centroid most.</p>
<p>2. Design greedy matching algorithm to do data association</p>
<p>To do this, we basically find all the nearest location pairs between 2 sequential frames, but we add a check: if the shortest distance is too far(eg, 100px), we don't regard it as a valid match and do next match computation.</p>
<p>We also tried simple greedy matching algorithm, which doesn't have a check. Unfortunately, the result is not satisfying</p>
<p>3. Use alpha/beta filter to predict next position(centroid) and velocity</p>
<p>We set alpha = beta = 1 to start and call greedy matching algorithm to do matching tasks. However, the matching pairs may not include all the objects since we put a check before matching. Those locations that are not matched will still be given an id but wait for future matching.</p>


<p>class: State: to store the object's centroid</p>
<p>class: Uncertain: to keep object locations that are not sure if they are spurious detections</p>
<p>class: helper: to implement some functions to calculate distances, draw dots on image and so on.</p>
<p>function: preprocess_data() to get the list of image frames and centroid of bats and cells</p>
<p>function: mainComputation() and my_motion_energy() to display the movements</p>
<p>class: DataAssociation: implement the greedy matching algorithm and store temporary unmatched object locations</p>
<p>class: ABFilter: init the alpha/beta filter and try to predict next frame locations and velocity.</p>

