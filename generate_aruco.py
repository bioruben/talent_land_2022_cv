import cv2 
import numpy as np
import argparse
  
# Project: ArUco Marker Generator
# Reference: https://www.pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/
 
# The different ArUco dictionaries built into the OpenCV library. 
ARUCO_DICT = {
  "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
  "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
  "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
  "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
  "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
  "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
  "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
  "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
  "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
  "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
  "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
  "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
  "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
  "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
  "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
  "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}
  

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Generate ArUco Marker')

  ### Positional arguments
  parser.add_argument('-id', '--id_aruco', default=10, type=int, help="Input Id for ArUco Marker")

  args = vars(parser.parse_args())

  aruco_marker_id  = (args["id_aruco"])

  desired_aruco_dictionary = "DICT_ARUCO_ORIGINAL"
  output_filename = "output/DICT_ARUCO_ORIGINAL_id_" + str(aruco_marker_id) + ".png"

  # Check if ArUco marker exist
  if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(
      args["type"]))
    sys.exit(0)
     
  # Load the ArUco dictionary
  this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[desired_aruco_dictionary])
   
  # Allocate memory for the ArUco marker
  print("[INFO] generating ArUCo tag type '{}' with ID '{}'".format(
    desired_aruco_dictionary, aruco_marker_id))
     
  # Create the ArUco marker and define size
  this_marker = np.zeros((300, 300, 1), dtype="uint8")
  cv2.aruco.drawMarker(this_aruco_dictionary, aruco_marker_id, 300, this_marker, 1)
   
  # Save the ArUco tag
  cv2.imwrite(output_filename, this_marker)
  cv2.imshow("ArUco Marker", this_marker)
  cv2.waitKey(0)