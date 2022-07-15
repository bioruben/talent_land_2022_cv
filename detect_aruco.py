import cv2
import numpy as np
import argparse
 
# Project: ArUco Marker Detector
# Reference: https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
 
desired_aruco_dictionary = "DICT_ARUCO_ORIGINAL"
 
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

  parser = argparse.ArgumentParser(description='Detect ArUco Marker')

  ### Positional arguments
  parser.add_argument('-p', '--pokemon_flag', default=False, help="Flag to draw traditional location of ArUco Markers or pokemons")

  args = vars(parser.parse_args())

  pokemon_flag  = (args["pokemon_flag"])

  # Check if ArUco marker exist.
  if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(
      args["type"]))
    sys.exit(0)
     
  # Load the ArUco dictionary
  print("[INFO] detecting '{}' markers...".format(
    desired_aruco_dictionary))
  this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[desired_aruco_dictionary])
  this_aruco_parameters = cv2.aruco.DetectorParameters_create()
   
  # Start the video stream
  # 0 is default camera, change value for different input camera
  cap = cv2.VideoCapture(0) 
   
  while(True):
  
    # Get Frame
    ret, frame = cap.read()
     
    # Detect ArUco markers in the video frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
      frame, this_aruco_dictionary, parameters=this_aruco_parameters)
       
    # Check that at least one ArUco marker was detected
    if len(corners) > 0:
      # Flatten the ArUco IDs list
      ids = ids.flatten()
       
      # Loop over the detected ArUco corners
      for (marker_corner, marker_id) in zip(corners, ids):
       
        # Extract the marker corners
        corners = marker_corner.reshape((4, 2))
        (top_left, top_right, bottom_right, bottom_left) = corners
         
        # Convert the (x,y) coordinate pairs to integers
        top_right = (int(top_right[0]), int(top_right[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        top_left = (int(top_left[0]), int(top_left[1]))      
         
        # Calculate the center of the ArUco marker
        center_x = int((top_left[0] + bottom_right[0]) / 2.0)
        center_y = int((top_left[1] + bottom_right[1]) / 2.0)
        
        # Draw pokemons over frame, else just draw bounding box for ArUco Markers
        if pokemon_flag:          
          width = abs(top_left[0] - bottom_right[0])
          height = abs(top_left[1] - bottom_right[1])
          dim = (width, height)

          try:
              bH, bW = frame.shape[:2]
              empty = 0 * np.ones((bH, bW, 3), dtype=np.uint8)
              # This drawing only consider one orientation of the marker
              if marker_id == 5:
                overlay_charmander = cv2.imread('data/charmander.png')
                overlay_charmander = cv2.resize(overlay_charmander, dim, interpolation = cv2.INTER_AREA)

                empty[:height, :width] = overlay_charmander
                _inp = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
                _out = np.float32([top_left, top_right, bottom_right, bottom_left])
                M = cv2.getPerspectiveTransform(_inp, _out)
                transformed = cv2.warpPerspective(empty, M, (bW, bH))
                frame[np.where(transformed != 0)] = transformed[np.where(transformed != 0)]

              elif marker_id == 1:
                overlay_bulbasaur = cv2.imread('data/bulbasaur.png')
                overlay_bulbasaur = cv2.resize(overlay_bulbasaur, dim, interpolation = cv2.INTER_AREA)

                empty[:height, :width] = overlay_bulbasaur
                _inp = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
                _out = np.float32([top_left, top_right, bottom_right, bottom_left])
                M = cv2.getPerspectiveTransform(_inp, _out)
                transformed = cv2.warpPerspective(empty, M, (bW, bH))
                frame[np.where(transformed != 0)] = transformed[np.where(transformed != 0)]

              elif marker_id == 10:
                overlay_pikachu = cv2.imread('data/pikachu.png')
                overlay_pikachu = cv2.resize(overlay_pikachu, dim, interpolation = cv2.INTER_AREA)

                empty[:height, :width] = overlay_pikachu
                _inp = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
                _out = np.float32([top_left, top_right, bottom_right, bottom_left])
                M = cv2.getPerspectiveTransform(_inp, _out)
                transformed = cv2.warpPerspective(empty, M, (bW, bH))
                frame[np.where(transformed != 0)] = transformed[np.where(transformed != 0)]

              elif marker_id == 2:
                overlay_squirtle = cv2.imread('data/squirtle.png')
                overlay_squirtle = cv2.resize(overlay_squirtle, dim, interpolation = cv2.INTER_AREA)

                empty[:height, :width] = overlay_squirtle
                _inp = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
                _out = np.float32([top_left, top_right, bottom_right, bottom_left])
                M = cv2.getPerspectiveTransform(_inp, _out)
                transformed = cv2.warpPerspective(empty, M, (bW, bH))
                frame[np.where(transformed != 0)] = transformed[np.where(transformed != 0)]

          except:
               print("Put the markers inside the scene")
        
        else:
          # Draw the bounding box of the ArUco detection and center
          cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
          cv2.line(frame, top_right, bottom_right, (255, 255, 255), 2)
          cv2.line(frame, bottom_right, bottom_left, (255, 0, 0), 2)
          cv2.line(frame, bottom_left, top_left, (0, 0, 255), 2)
          cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
         
          # Draw the ArUco marker ID on the video frame
          # The ID is always located at the top_left of the ArUco marker
          cv2.putText(frame, str(marker_id), 
            (top_left[0], top_left[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2)
  
    # Display the resulting frame
    cv2.imshow('frame',frame)
          
    # If "q" is pressed on the keyboard, 
    # exit this loop
    k = cv2.waitKey(1)
    if k==27:
      break
  
  # Close down the video stream
  cap.release()
  cv2.destroyAllWindows()
