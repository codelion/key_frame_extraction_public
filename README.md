# Project:

This key frame extraction project is to extract all the unique and high quality key frames (images) from a video.

# Requirements:

All the required libraries are mentioned in requirement.txt file. 

Use pip install -r requirement.txt to install all the requirements.

# How to run the code:

To run the code, execute below command with all the required parameters. 

python candidate_frames_folder.py --input_videos "sample_video.mp4" --output_folder_video_image candidate_frames_and_their_cluster_folder --output_folder_video_final_image final_images

This command will create a new folder with the same name as input video name and inside that folder, candidate frames and their clusters based on similarity will be created in "candidate_frames_and_their_cluster_folder" and final key frames in "final_images" folder respectively

On Jetson run as follows:

`export LD_PRELOAD=/home/securade/key_frame_extraction_public/.venv/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0`

`python candidate_frames_folder.py --input_videos ../edge-app/examples/video_1.mp4 --output_folder_video_image tmp-frames --output_folder_video_final_image final-frames`

