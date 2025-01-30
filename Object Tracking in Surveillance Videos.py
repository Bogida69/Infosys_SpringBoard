import streamlit as st
import cv2
import numpy as np
import math
import os
from object_detection import ObjectDetection
import tempfile

def run_tracking_on_video(uploaded_video, weights_path, cfg_path, classes_path, frames_per_second, min_probability):
    # Save the uploaded video to a temporary file
    temp_video_path = tempfile.mktemp(suffix=".mp4")

    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Initialize Object Detection
    od = ObjectDetection(weights_path=weights_path, cfg_path=cfg_path)

    # Open video file
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {temp_video_path}")

    # Hardcoded mask (change the path to your mask image)
    mask_path = "images\mask.png"  # Replace this with your mask image path
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        raise IOError(f"Failed to load mask image: {mask_path}")

    # Get video properties for saving
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Ensure the output directory exists
    output_dir = r"C:\Drive E\me\Python_for_ML\Infosys\Infosys_SpringBoard\Streamlit_Output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the output path for the processed video
    output_path = os.path.join(output_dir, "Streamlit_Video.avi")

    # Initialize video writer with the correct path
    output_video = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height)
    )

    # Get total number of frames to process
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the progress bar
    progress_bar = st.progress(0)

    # Initialize placeholder for the percentage text
    progress_text = st.empty()

    # Initialize variables for tracking
    count = 0
    center_points_prev_frame = []
    tracking_objects = {}
    track_id = 0

    while True:
        ret, frame = cap.read()
        count += 1
        if not ret:
            break

        # Apply the mask to the frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Store center points of the current frame
        center_points_cur_frame = []

        # Detect objects on the frame
        try:
            class_ids, scores, boxes = od.detect(masked_frame)
        except Exception as e:
            st.error(f"Error during object detection: {e}")
            break

        for box in boxes:
            x, y, w, h = box

            # Calculate center points of the bounding box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))

            # Draw rectangle around detected object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update tracking objects
        for pt in center_points_cur_frame:
            same_object_detected = False
            for object_id, prev_pt in tracking_objects.items():
                distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])

                if distance < 35:  # Threshold distance
                    tracking_objects[object_id] = pt
                    same_object_detected = True
                    break

            # Assign new ID to new object
            if not same_object_detected:
                tracking_objects[track_id] = pt
                track_id += 1

        # Draw tracking points and IDs
        for object_id, pt in tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Ensure numbers are drawn last, on top of all objects
            cv2.putText(
                frame,
                str(object_id),
                (pt[0] - 10, pt[1] - 10),  # Offset for better visibility
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text for visibility
                2,  # Thickness
                lineType=cv2.LINE_AA,  # Anti-aliased text
            )

        # Write the processed frame to the output video
        output_video.write(frame)

        # Update progress bar
        progress = count / total_frames
        progress_bar.progress(progress)

        # Update the percentage text
        progress_text.text(f"Processing: {int(progress * 100)}%")

        # Prepare for next frame
        center_points_prev_frame = center_points_cur_frame.copy()

    cap.release()
    output_video.release()

    return output_path

def main():
    st.title("Object Tracking in Surveillance Videos")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    frames_per_second = st.sidebar.slider("Frames per second for processing", 1, 30, 10)
    min_probability = st.sidebar.slider("Minimum probability for detection (%)", 10, 100, 30)

    # File uploader for video only (mask is hardcoded)
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    # File paths for YOLO model
    weights_path = r"models\dnn_model\yolov4.weights"
    cfg_path = r"models\dnn_model\yolov4.cfg"
    classes_path = r"models\dnn_model\classes.txt"

    if uploaded_video:
        run_button = st.button("Run Object Tracking")
        if run_button:
            st.write("Running object tracking...")

            try:
                output_video_path = run_tracking_on_video(
                    uploaded_video, weights_path, cfg_path, classes_path, frames_per_second, min_probability
                )

                st.success("Object tracking completed!")

                # Display the download button next to the "Run Object Tracking" button
                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f,
                        file_name="Streamlit_Video.avi",
                        mime="video/avi"
                    )

            except Exception as e:
                st.error(f"Error during object tracking: {e}")
    else:
        st.warning("Please upload a video file.")

if __name__ == "__main__":
    main()
