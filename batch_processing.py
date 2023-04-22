import cv2
import os
from tqdm.auto import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import repeat

'''
This module is designed primarily for batch processing MOV files stored in one folder using multi-threading. It contains the following features, and more details can be found in individual function documentation:

1. grayscale: convert the current coloured video to grayscale
2. crop: crop the width of grayscle videos to 2200
3. thresholding: threshold the video by using the adaptive Gaussian thresholding in opencv
4. canny_edge_detection: detect the edge by using the canny thresholding
5. auto_canny_edge_detection: avoid find-tuning of maximum and minimum value in cv2.Canny and detect the edge based on the statistics
6. multi-threading: process all the videos together by using multi-threading 

---
The global variables that have to be defined before using this module:
video_folder_path: the local path to the folders that contains all mov files
---
'''


###################################################################################

# non-essential feature from this module
# create results folders if not exist already

def check_create_folders(path):
    if not os.path.exists(path):
        os.makedirs(path)



###################################################################################

def multi_threading(video_folder_path, function_name):
	'''
	This function runs multi threading to achieve batch processing. 
	--- input
	video_folder_path: the local path to the folders that contains all mov files
	function_name: the function that needs to be processed
	'''

	mov_filenames = [f for f in os.listdir(video_folder_path) if ( f.endswith('.MOV') or f.endswith('.mov') )]

	with tqdm(total=len(mov_filenames)) as pbar:
		with ThreadPoolExecutor(max_workers=len(mov_filenames)) as ex:
			futures = [ex.submit(function_name, video_folder_path, filename) for filename in mov_filenames]
			for future in as_completed(futures):
				pbar.update(1)



def grayscale(video_folder_path, filename):

	'''
	This function converts videos to grayscale frame by frame and stores the videos in 'grayscale' folder.
	'''
	
	# load the current video to process
	vid = cv2.VideoCapture(os.path.join(video_folder_path, filename))
	frame_width = int(vid.get(3))
	frame_height = int(vid.get(4))
	size = (frame_width, frame_height)

	video_gray_folder_path = video_folder_path + '/grayscale'
	check_create_folders(video_gray_folder_path)

    # create the threshold video in a separate folder called 'threshold'
	vid_gray = cv2.VideoWriter(f'{os.path.join(video_gray_folder_path, filename[:-4])}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60, size, 0) 

    # get the total number of frames in the video
	frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # running the loop
	with tqdm(total=frames, leave=False) as pbar:
        
		for j in range(frames):
            
            # extracting the frames 
			ret, img = vid.read() 

            # converting to gray-scale 
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # write to threshold
			vid_gray.write(gray)

            # displaying the video 
            # cv2.imshow("Live", thres)
            
			pbar.update(1)
    
            # exiting the loop 
			key = cv2.waitKey(1)
			if key == ord("q"): 
				break
			else:
				continue


    # When everything done, release the video capture object
	vid.release()
	vid_gray.release()

    # closing all frames
	cv2.destroyAllWindows() 






def crop(video_folder_path, filename):

	'''
	This function crops the width to 2200 pixels specifically for a type of bright field microscope
	'''
    
    # load the current video to process
	vid = cv2.VideoCapture(os.path.join(video_folder_path, filename))
	frame_width = int(vid.get(3)) 
	frame_height = int(vid.get(4))
	size = (frame_width, frame_height)

	video_rect_crop_folder_path = video_folder_path + '/cropped'
	check_create_folders(video_rect_crop_folder_path)

    # create the threshold video in a separate folder called 'threshold'
	vid_crop = cv2.VideoWriter(f'{os.path.join(video_rect_crop_folder_path, filename[:-4])}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60, size, 0) 

    # get the total number of frames in the video
	frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # running the loop
	with tqdm(total=frames, leave=False) as pbar:
        
		for j in range(frames):
            
            # extracting the frames 
			ret, img = vid.read() 

            # converting to gray-scale 
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # just cut it to smaller rectangle igonring the contours
			w = 2200
			h = int(gray.shape[0])
			offset = 20

			x = int( (gray.shape[1] - w )/2 ) - offset
			y = 0

			crop = gray[y:y+h, x:x+w]

            # write to threshold
			vid_crop.write(crop)

            # displaying the video 
            # cv2.imshow("Live", thres)
            
			pbar.update(1)
    
            # exiting the loop 
			key = cv2.waitKey(1)
			if key == ord("q"): 
				break
			else:
				continue


    # When everything done, release the video capture object
	vid.release()
	vid_crop.release()

    # closing all frames
	cv2.destroyAllWindows() 




def thresholding(video_folder_path, filename):

	'''
	This function uses adaptive Gaussian thresholding in opencv to threshold the videos.
	'''
    
    # load the current video to process
	vid = cv2.VideoCapture(os.path.join(video_folder_path, filename))
	frame_width = int(vid.get(3)) 
	frame_height = int(vid.get(4))
	size = (frame_width, frame_height)

	video_threshold_folder_path = video_folder_path + '/threshold'
	check_create_folders(video_threshold_folder_path)


    # create the threshold video in a separate folder called 'threshold'
	vid_threshold = cv2.VideoWriter(f'{os.path.join(video_threshold_folder_path, filename[:-4])}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60, size, 0) 

    # get the total number of frames in the video
	frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # running the loop
	with tqdm(total=frames, leave=False) as pbar:
        
		for j in range(frames):
            
            # extracting the frames 
			ret, img = vid.read() 

            # converting to gray-scale 
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            # threshold 
			thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
				cv2.THRESH_BINARY,11,2)

            # write to threshold
			vid_threshold.write(thres)

            # displaying the video 
            # cv2.imshow("Live", thres)
            
			pbar.update(1)
    
            # exiting the loop 
			key = cv2.waitKey(1)
			if key == ord("q"): 
				break
			else:
				continue


    # When everything done, release the video capture object
	vid.release()
	vid_threshold.release()

    # closing all frames
	cv2.destroyAllWindows() 










def canny_edge_detection(video_folder_path, filename):

	''' 
	This functions detects the edge by using the Canny filter in openCV. The minimum and maximum values for the Canny edge detection algorithm were set to be(100, 200).
	'''
    
    # load the current video to process
	vid = cv2.VideoCapture(os.path.join(video_folder_path, filename))
	frame_width = int(vid.get(3)) 
	frame_height = int(vid.get(4))
	size = (frame_width, frame_height)

	video_canny_folder_path = video_folder_path + '/canny-edge-detection'
	check_create_folders(video_canny_folder_path)

    # create the threshold video in a separate folder called 'threshold'
	vid_threshold = cv2.VideoWriter(f'{os.path.join(video_canny_folder_path, filename[:-4])}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60, size, 0) 

    # get the total number of frames in the video
	frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # running the loop
	with tqdm(total=frames, leave=False) as pbar:
        
		for j in range(frames):
            
            # extracting the frames 
			ret, img = vid.read() 

            # converting to gray-scale 
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            # canny threshold 
			edges = cv2.Canny(gray, 100, 200)

            # write to threshold
			vid_threshold.write(edges)

            # displaying the video 
            # cv2.imshow("Live", thres)
            
			pbar.update(1)
    
            # exiting the loop 
			key = cv2.waitKey(1)
			if key == ord("q"): 
				break
			else:
				continue


    # When everything done, release the video capture object
	vid.release()
	vid_threshold.release()

    # closing all frames
	cv2.destroyAllWindows() 







def auto_canny_edge_detection(video_folder_path, filename):
	'''
	This function automatically calculates the most appropriate minimum and maximum values for Canny edge detection algorithm for each frame.
	'''
    
    # load the current video to process
	vid = cv2.VideoCapture(os.path.join(video_folder_path, filename))
	frame_width = int(vid.get(3)) 
	frame_height = int(vid.get(4))
	size = (frame_width, frame_height)


	video_auto_canny_folder_path = video_folder_path + '/auto-canny-edge-detection'
	check_create_folders(video_auto_canny_folder_path)

    # create the threshold video in a separate folder called 'threshold'
	vid_threshold = cv2.VideoWriter(f'{os.path.join(video_auto_canny_folder_path, filename[:-4])}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60, size, 0) 

    # get the total number of frames in the video
	frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # running the loop
	with tqdm(total=frames, leave=False) as pbar:
        
		for j in range(frames):
            
            # extracting the frames 
			ret, img = vid.read() 

            # converting to gray-scale 
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            # canny threshold 
			v = np.median(gray)
    
            # apply automatic Canny edge detection using the computed median
			sigma = 0.33
			lower = int(max(0, (1.0 - sigma) * v))
			upper = int(min(255, (1.0 + sigma) * v))
    
			edges = cv2.Canny(gray, lower, upper)

            # write to threshold
			vid_threshold.write(edges)

            # displaying the video 
            # cv2.imshow("Live", thres)
            
			pbar.update(1)
    
            # exiting the loop 
			key = cv2.waitKey(1)
			if key == ord("q"): 
				break
			else:
				continue


    # When everything done, release the video capture object
	vid.release()
	vid_threshold.release()

    # closing all frames
	cv2.destroyAllWindows() 
