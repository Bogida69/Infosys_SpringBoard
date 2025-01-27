from imageai.Detection import VideoObjectDetection
import os
import time
import torch
import tensorflow as tf

# Start timing
start_time = time.time()

# Set the execution path
execution_path = os.getcwd()

# Define the functions for handling frame, second, and minute output
def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME ", frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")

# Initialize the video detector
video_detector = VideoObjectDetection()

# Set model type and path
video_detector.setModelTypeAsTinyYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, r"C:\Drive E\me\Python_for_ML\Personal_Projects\HMPV\Face_Mask_Dataset\Result\runs\detect\train\weights\best.pt"))
video_detector.loadModel()

# Define the output file name for the processed video
output_video_file_path = os.path.join(execution_path, r"C:\Drive E\me\Python_for_ML\Personal_Projects\HMPV\Face_Mask_Dataset\Mask OUtput.mp4")

# Process the video and save it to the specified output file
video_detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, r"C:\Drive E\me\Python_for_ML\Personal_Projects\HMPV\Face_Mask_Dataset\Mask_VIdeo.mp4"),
    output_file_path=output_video_file_path,  # Path to save the processed video
    frames_per_second=10,
    per_second_function=forSeconds,
    per_frame_function=forFrame,
    per_minute_function=forMinute,
    minimum_percentage_probability=30
)

# End timing and calculate the duration
end_time = time.time()
execution_duration = end_time - start_time

# Print the duration of execution
print("Time taken to run the code:", execution_duration, "seconds")
