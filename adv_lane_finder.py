from moviepy.editor import VideoFileClip

import FindLanes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


fl = FindLanes.FindLanes()

white_output = 'output_images/project_video_solution.mp4'


## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip("project_video.mp4")

# clip1.save_frame("frame.png", t=41.86) # saves the frame a t=2s

# clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(fl) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

