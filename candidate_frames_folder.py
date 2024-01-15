import time
start_time = time.time()
import argparse
import os
import sys
import os.path
import cv2
import extracting_candidate_frames
import clustering_with_hdbscan
import json
# from multiprocessing import Pool, Process, cpu_count
import logging


logging.basicConfig(filename='key_frames.log',format='%(asctime)s  %(levelname)s:%(message)s',level=logging.DEBUG)
logging.info('---------------------------------------------------------------------------------------------------------')

"""# Running the code 
 python candidate_frames_folder.py --input_videos sample_video.mp4 --output_folder_video_image candidate_frames_and_their_cluster_folder /
 --output_folder_video_final_image final_images"""


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_videos",
        help="Path to the input video file"
    )

    # Required arguments: output candidate images of video file.
    parser.add_argument(
        "--output_folder_video_image",
        help="folder for candidates frames"
    )

    # Required arguments: output candidate final images of video file..
    parser.add_argument(
        "--output_folder_video_final_image",
        help="Folder for key frames to be saved."
    )

    args = parser.parse_args()
    logging.info('file execution started for input video {}'.format(args.input_videos))
    vd = extracting_candidate_frames.FrameExtractor()
    if not os.path.isdir(args.input_videos.rsplit( ".", 1 )[ 0 ]):
        os.makedirs(args.input_videos.rsplit( ".", 1 )[ 0 ] + '/' + args.output_folder_video_image)
        os.makedirs(args.input_videos.rsplit( ".", 1 )[ 0 ] + '/' + args.output_folder_video_final_image)
    imgs=vd.extract_candidate_frames(args.input_videos)
    # data to be written
    output = { "total_frames" : vd.input_frames}
    output.update({"candidate_frames" : len(imgs)})

    for counter, img in enumerate(imgs):
        vd.save_frame_to_disk(
            img,
            file_path=os.path.join(args.input_videos.rsplit( ".", 1 )[ 0 ],args.output_folder_video_image),
            file_name="test_" + str(counter),
            file_ext=".jpeg",
        )
    final_images = clustering_with_hdbscan.ImageSelector()
    imgs_final = final_images.select_best_frames(imgs,os.path.join(args.input_videos.rsplit( ".", 1 )[ 0 ],args.output_folder_video_image))
    
    output.update({"key_frames" : len(imgs_final)})
    output.update({"percentage" : len(imgs_final)/len(imgs)*100})
    
    # frame = cv2.imread(imgs_final[0])
    height, width, layers = imgs_final[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    video = cv2.VideoWriter(args.input_videos + "-processed.mp4", fourcc, 20.0, (width,height))

    for counter, i in enumerate(imgs_final):
        vd.save_frame_to_disk(
            i,
            file_path=os.path.join(args.input_videos.rsplit( ".", 1 )[ 0 ],args.output_folder_video_final_image),
            file_name="test_" + str(counter),
            file_ext=".jpeg",
        )
        video.write(i)
    video.release()
    # Serializing json
    json_object = json.dumps(output, indent=4)
        
    # Writing to sample.json
    with open(args.input_videos+"-output.json", "w") as outfile:
        outfile.write(json_object) 
    logging.info("--- {a} seconds to extract key frames from {b}---".format(a= (time.time() - start_time),b = args.input_videos))

if __name__ == "__main__":
    main(sys.argv)