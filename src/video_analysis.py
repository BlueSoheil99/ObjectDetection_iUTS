import copy
from collections import defaultdict
import os

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

# class VideoAnalyzer:
#     def __init__(self, video_dir:str, output_dir:str):
#         self.video_dir = video_dir
#         self.output_dir = output_dir
#         # Make sure output dir exists
#         os.makedirs(self.output_dir, exist_ok=True)
#
#         # analyze video
#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         resolution = (int(cap.get(3)), int(cap.get(4)))

def video_analysis(video_path:str, model:str, output_path:str, id=0, acc=3):
    """
    :param video_path: the address of the input video
    :param model: path of the model to be used
    :param output_path: where output will be saved
    :param id: output file id
    :param acc: decimal place accuracy
    :return:
    """
    # Load model
    model = YOLO(model)
    # Make sure output dir exists
    os.makedirs(output_path, exist_ok=True)

    # analyze video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    resolution = (int(cap.get(3)), int(cap.get(4)))

    # prepare output mp4
    out_mp4 = cv2.VideoWriter(f'{output_path}/{str(id)}_full_out.mp4',
                              cv2.VideoWriter_fourcc(*"XVID"),
                              fps, resolution)

    # Store the track history for annotations
    track_history = defaultdict(lambda: [])
    # store result history (frame, results) in output
    results = []
    # store data for csv output
    time_index = 0 # the current frame (1 to infinity)
    track_data = defaultdict(lambda: defaultdict(lambda: ""))
    # data is stored in this format {track_id:(time_index, x, y), ...}

    # analysis
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        time_index += 1

        if success:
            # Run YOLO11 tracking on the frame with tracking
            result = model.track(frame, persist=True)[0]
            results.append((frame, result))

            # Get the boxes and track IDs
            if result.boxes and result.boxes.is_track:
                # plot the boxes + ids + classes
                frame = result.plot(True, 1, None)

                # Plot the tracks
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()
                for i in range(0, len(boxes)):
                    x, y, w, h = boxes[i]
                    track = track_history[track_ids[i]]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # save data for dataframe exportation
                    track_t = track_data[track_ids[i]]
                    track_t["Class"] = result.names[int(result[i].boxes.cls.tolist()[0])]
                    track_t[round(time_index / fps, acc)] = str(round(float(x),acc)) + ", " + str(round(float(y),acc))
                    # change acc input for more specific timings

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

            # write to the videowriter
            out_mp4.write(frame)
        else:
            # Break the loop if the end of the video is reached
            break

    # release captures
    cap.release()
    out_mp4.release()

    # Convert the dictionary to dataframe
    # 'orient='index' tells pandas to use the outer dictionary keys as the DataFrame index (rows)
    df = pd.DataFrame.from_dict(track_data, orient='index')
    df.index.name = 'ID'

    # save data as csv file
    output_filename = f'{output_path}/{str(id)}_output_data.csv'
    df.to_csv(output_filename)

    return (df, results, (fps, resolution))


def specific_output(output_dir:str, requested_ids:list, results, fps_resolution:list, id=0):
    """

    :param requested_ids:
    :param results: a results object from ultralytics
    :param fps_resolution:
    :param id:
    :return:
    """
    # potentially unecessary, this just ensures we aren't modifying the actual results object
    resultss = copy.deepcopy(results)
    # create the writing object
    name = f"{output_dir}/{str(id)}_collision_{str(requested_ids[0])}_{str(requested_ids[1])}.mp4"
    out_mp4 = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"XVID"), fps_resolution[0], fps_resolution[1])

    # Iterate through the results (one result object per frame)
    for r in resultss:
        if r[1].boxes is not None and r[1].boxes.id is not None:
            boxes = r[1].boxes
            track_ids = boxes.id.int().cpu().tolist() # Convert to list of integers for easier filtering
            desired_track_ids = requested_ids

            # Create a boolean mask to filter based on desired_track_ids
            # Ensure track_ids are handled as a tensor for comparison with .data
            mask_track_id = torch.tensor([tid in desired_track_ids for tid in track_ids], device=boxes.data.device)

            # Apply the mask to the entire box data (which includes coordinates, confidence, class, and ID)
            filtered_box_data = boxes.data[mask_track_id]

            # Update the .boxes attribute with the filtered data
            r[1].boxes = Boxes(filtered_box_data, orig_shape=r[1].orig_shape)

            # plot
            annotated_frame = r[1].plot()
            
            out_mp4.write(annotated_frame)
        else:
            # Handle cases where no detections or track IDs are present in a frame
            print("No detections or track IDs in this frame.")
    
    out_mp4.release()
