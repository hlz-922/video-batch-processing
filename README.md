# video_batch_processing

This module is designed primarily for batch processing MOV files stored in one folder using multi-threading. It contains the following features, and more details can be found in individual function documentation:

1. grayscale: convert the current coloured video to grayscale
2. crop: crop the width of grayscle videos to 2200
3. thresholding: threshold the video by using the adaptive Gaussian thresholding in opencv
4. canny_edge_detection: detect the edge by using the canny thresholding
5. auto_canny_edge_detection: avoid find-tuning of maximum and minimum value in cv2.Canny and detect the edge based on the statistics
6. multi-threading: process all the videos together by using multi-threading 

---
The global variable that has to be defined before using this module:
video_folder_path: the local path to the folders that contains all mov files
---
