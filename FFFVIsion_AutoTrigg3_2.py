# Import required libraries
from sys import exit
import numpy as np
import cv2
print('OpenCV version: '+cv2.__version__)
import Jetson.GPIO as GPIO
from time import sleep
import datetime
from signal import signal, SIGINT
from sys import exit
import pandas as pd

# %matplotlib qt5

################ ALL FUNCTIONS DEFINITIONS ################

# define function to obtain grayscale perspective corrected view


def perspCorrection(img, pt1, pt2, pt3, pt4, scale_width, scale_height):

    # Create a copy of the image
    img_copy = np.copy(img)

    # Convert to RGB so as to display via matplotlib
    # Using Matplotlib we can easily find the coordinates of the 4 points that is essential for finding then transformation matrix
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # to calculate the transformation matrix
    input_pts = np.float32([pt1, pt2, pt3, pt4])
    output_pts = np.float32(
        [[0, 0], [scale_width-1, 0], [0, scale_height-1], [scale_width-1, scale_height-1]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    # Apply the perspective transformation to the image
    # ,flags=cv2.INTER_LINEAR) cv2.INTER_CUBIC is also an option
    imgPersp = cv2.warpPerspective(img, M, (scale_width, scale_height))
    imgGrayPersp = cv2.cvtColor(imgPersp, cv2.COLOR_BGR2GRAY)

    # visulaize corners using cv2 circles
    for x in range(0, 4):
        cv2.circle(
            img_copy, (input_pts[x][0], input_pts[x][1]), 5, (0, 0, 255), cv2.FILLED)

    return [img_copy, imgPersp, imgGrayPersp]


def extractTopBottom(img, tStart, tEnd, bStart, bEnd):
    img_top = img[tStart[1]:tEnd[1], tStart[0]:tEnd[0]]
    img_bottom = img[bStart[1]:bEnd[1], bStart[0]:bEnd[0]]

    return [img_top, img_bottom]


def gaussianBlur(img, fsize):

    # gaussian blur
    gblur = cv2.GaussianBlur(img, (fsize, fsize), 0)

    return gblur


def medianBlur(img, fsize=3):

    # median blur - effective at removing salt and pepper noise
    mblur = cv2.medianBlur(img, fsize)

    return mblur


def bilateralFilter(img):

    # Bilateral filter preserves edges while removing noise
    bfblur = cv2.bilateralFilter(img, 9, 75, 75)

    return bfblur


def gAdaptiveThresholding(img):

    # median filtering
    adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)

    return adaptive_gaussian


def morphOps(img, kernel1, kernel2):

    # Closing = Dilation + Erosion
    # dilation
    mask_dil = cv2.dilate(img, kernel1, iterations=2)

    # erosion
    mask_erode = cv2.erode(mask_dil, kernel2, iterations=1)

    return mask_erode


def computeW(img):

    avg_num_pixels = 159
    scaling_factor = 1.0
    mm_per_pixel = ((1/32)*25.4)/(scaling_factor*avg_num_pixels)
    edge_length_threshold = 45

    # Predefine arrays for data storage
    approx_edges = 5
    num_edges = np.zeros(img.shape[0])  # ,dtype=np.uint16)
    edge_start = np.zeros([img.shape[0], approx_edges])  # ,dtype=np.uint16)
    edge_end = np.zeros([img.shape[0], approx_edges])  # ,dtype=np.uint16)

    edge_count = 0
    k = 0

    # start scanning from (0,0) until black pixel is found
    # go across columns first

    for i in range(img.shape[0]):

        found_edge = False
        temp_edge_count = 0
        k = 0

        for j in range(img.shape[1]):

            if(img[i, j] <= 50):
                # Black pixel found - edge
                if(found_edge == False):
                    found_edge = True
                    temp_edge_count += 1
                    num_edges[i] = temp_edge_count
                    edge_start[i][k] = j
                    k += 1

            else:
                if(found_edge):
                    edge_end[i][k-1] = j-1
                    found_edge = False

    if max(num_edges) == 2:

        line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
            num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
        line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
            num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

        line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
        line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

        edge_count = 2
        case_cond = 1

    elif max(num_edges) == 3:

        # 3 edges
        # Test for 3
        bool_edge_loc = num_edges == 3
        edge_px_len = np.diff(np.where(np.concatenate(
            ([bool_edge_loc[0]], bool_edge_loc[:-1] != bool_edge_loc[1:], [True])))[0])[::2]
        true2edge = edge_start[:, 1][np.argwhere(num_edges == 3)]
        true2edge = true2edge[true2edge < 250]

        # check location of occurance of 2nd significant edge
        if np.mean(true2edge) < 250:

            if len(np.unique(num_edges)) > 1:

                # More than two significant edges found but edge is NOT continuous
                if len(true2edge) >= edge_length_threshold:
                    # significant 2nd edge found
                    # use this for road width calculation

                    line1_start = (round(np.mean((edge_start[:, 1][np.argwhere(num_edges == 3)][edge_start[:, 1][np.argwhere(
                        num_edges == 3)] < 250] + edge_end[:, 1][np.argwhere(num_edges == 3)][edge_end[:, 1][np.argwhere(num_edges == 3)] < 250])/2)), 0)
                    line1_end = (round(np.mean((edge_start[:, 1][np.argwhere(num_edges == 3)][edge_start[:, 1][np.argwhere(
                        num_edges == 3)] < 250] + edge_end[:, 1][np.argwhere(num_edges == 3)][edge_end[:, 1][np.argwhere(num_edges == 3)] < 250])/2)), img.shape[0])

                    line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                        num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
                    line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                        num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                    edge_count = 3
                    case_cond = 2

                else:
                    # treat this as stray edge - not significant
                    line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
                        num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
                    line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
                        num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                    line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                        num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
                    line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                        num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                    edge_count = 2
                    case_cond = 3

            elif np.logical_and(len(np.unique(num_edges)) == 1, np.unique(num_edges)[0] == 3):

                # 3 significant edges found AND edges are continuous
                line1_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 3)])/2)), 0)
                line1_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 3)])/2)), img.shape[0])

                line2_start = (round(np.mean((edge_start[:, 2][np.argwhere(
                    num_edges == 3)] + edge_end[:, 2][np.argwhere(num_edges == 3)])/2)), 0)
                line2_end = (round(np.mean((edge_start[:, 2][np.argwhere(
                    num_edges == 3)] + edge_end[:, 2][np.argwhere(num_edges == 3)])/2)), img.shape[0])

                edge_count = 3
                case_cond = 4

        elif np.logical_and(np.mean(edge_start[:, 1][np.argwhere(num_edges == 3)]) > 250, np.mean(edge_start[:, 1][np.argwhere(num_edges == 3)]) < 510):

            # second edge found before right significant edge
            # check for significant length

            if np.sum(edge_px_len) >= edge_length_threshold:

                # second edge is a valid edge
                line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
                    num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
                line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
                    num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 3)])/2)), 0)
                line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 3)])/2)), img.shape[0])

                edge_count = 3
                case_cond = 5

            else:

                # second edge is just noise
                line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
                    num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
                line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
                    num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
                line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                edge_count = 2
                case_cond = 6

        else:

            # A third stray edge found beyond road 2nd edge to the right - noise
            line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
                num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
            line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
                num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

            line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
            line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

            edge_count = 3
            case_cond = 7

    elif max(num_edges) == 4:

        # 4 edges
        # Test for 4
        bool_edge_loc = num_edges == 4
        edge_px_len = np.diff(np.where(np.concatenate(
            ([bool_edge_loc[0]], bool_edge_loc[:-1] != bool_edge_loc[1:], [True])))[0])[::2]

        # check location of occurance of 2nd significant edge
        if np.mean(edge_start[:, 1][np.argwhere(num_edges == 4)]) < 250:

            if len(np.unique(num_edges)) > 1:

                # check location of 4th edge
                if np.mean(edge_start[:, 3][np.argwhere(num_edges == 4)]) < 510:

                    if np.logical_and(len(edge_start[:, 1][np.argwhere(num_edges == 4)]) < edge_length_threshold, len(edge_start[:, 1][np.argwhere(num_edges == 3)]) >= edge_length_threshold):
                        # edge before right significant edge is not significant

                        line1_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 4)] + edge_end[:, 1][np.argwhere(num_edges == 4)])/2)), 0)
                        line1_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 4)] + edge_end[:, 1][np.argwhere(num_edges == 4)])/2)), img.shape[0])

                        line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
                        line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                        edge_count = 3
                        case_cond = 8

                    elif np.logical_and(len(edge_start[:, 1][np.argwhere(num_edges == 4)]) >= edge_length_threshold, len(edge_start[:, 2][np.argwhere(num_edges == 4)]) >= edge_length_threshold):

                        line1_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 4)] + edge_end[:, 1][np.argwhere(num_edges == 4)])/2)), 0)
                        line1_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 4)] + edge_end[:, 1][np.argwhere(num_edges == 4)])/2)), img.shape[0])

                        line2_start = (round(np.mean((edge_start[:, 2][np.argwhere(
                            num_edges == 4)] + edge_end[:, 2][np.argwhere(num_edges == 4)])/2)), 0)
                        line2_end = (round(np.mean((edge_start[:, 2][np.argwhere(
                            num_edges == 4)] + edge_end[:, 2][np.argwhere(num_edges == 4)])/2)), img.shape[0])

                        edge_count = 4
                        case_cond = 9

                    elif np.logical_and(len(edge_start[:, 1][np.argwhere(num_edges == 4)]) >= edge_length_threshold, len(edge_start[:, 2][np.argwhere(num_edges == 4)]) < edge_length_threshold):

                        line1_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 3)])/2)), 0)
                        line1_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 3)])/2)), img.shape[0])

                        line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
                        line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                        edge_count = 3
                        case_cond = 10

                    elif np.logical_and(len(edge_start[:, 1][np.argwhere(num_edges == 4)]) < edge_length_threshold, len(edge_start[:, 2][np.argwhere(num_edges == 4)]) >= edge_length_threshold):

                        line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
                            num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
                        line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
                            num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                        line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 3)])/2)), 0)
                        line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 3)])/2)), img.shape[0])

                        edge_count = 3
                        case_cond = 11

                    else:

                        line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
                            num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
                        line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
                            num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                        line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
                        line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                        edge_count = 2
                        case_cond = 12

                else:

                    if len(edge_start[:, 1][np.argwhere(num_edges == 3)]) >= edge_length_threshold:

                        line1_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 3)])/2)), 0)
                        line1_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 3)] + edge_end[:, 1][np.argwhere(num_edges == 3)])/2)), img.shape[0])

                        line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
                        line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                        edge_count = 3
                        case_cond = 13

                    else:

                        line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
                            num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
                        line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
                            num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                        line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
                        line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                        edge_count = 2
                        case_cond = 14

            elif np.logical_and(len(np.unique(num_edges)) == 1, np.unique(num_edges)[0] == 4):

                # 4 significant edges found AND edges are continuous
                line1_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 4)] + edge_end[:, 1][np.argwhere(num_edges == 4)])/2)), 0)
                line1_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 4)] + edge_end[:, 1][np.argwhere(num_edges == 4)])/2)), img.shape[0])

                line2_start = (round(np.mean((edge_start[:, 2][np.argwhere(
                    num_edges == 4)] + edge_end[:, 2][np.argwhere(num_edges == 4)])/2)), 0)
                line2_end = (round(np.mean((edge_start[:, 2][np.argwhere(
                    num_edges == 4)] + edge_end[:, 2][np.argwhere(num_edges == 4)])/2)), img.shape[0])

                edge_count = 4
                case_cond = 15

        elif np.logical_and(np.mean(edge_start[:, 1][np.argwhere(num_edges == 4)]) > 250, np.mean(edge_start[:, 1][np.argwhere(num_edges == 4)]) < 510):

            # second edge found before right significant edge
            # Check for significant length

            if len(edge_start[:, 1][np.argwhere(num_edges == 4)]) >= edge_length_threshold:

                # second edge is a valid edge
                line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
                    num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
                line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
                    num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 4)] + edge_end[:, 1][np.argwhere(num_edges == 4)])/2)), 0)
                line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 4)] + edge_end[:, 1][np.argwhere(num_edges == 4)])/2)), img.shape[0])

                edge_count = 3
                case_cond = 16

            else:

                # 3rd and 4th edges are beyond right significant edge and treat as noise
                line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
                    num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
                line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
                    num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
                line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                    num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

                edge_count = 2
                case_cond = 17

        else:

            line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
                num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
            line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
                num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

            line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
                num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
            line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
                num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

            edge_count = 2
            case_cond = 18

    else:

        line1_start = (round(np.mean((edge_start[:, 0][np.argwhere(
            num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), 0)
        line1_end = (round(np.mean((edge_start[:, 0][np.argwhere(
            num_edges == 2)] + edge_end[:, 0][np.argwhere(num_edges == 2)])/2)), img.shape[0])

        line2_start = (round(np.mean((edge_start[:, 1][np.argwhere(
            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), 0)
        line2_end = (round(np.mean((edge_start[:, 1][np.argwhere(
            num_edges == 2)] + edge_end[:, 1][np.argwhere(num_edges == 2)])/2)), img.shape[0])

        edge_count = 2
        case_cond = 19

    # convert to BGR image and draw line
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # computed road width in pixels
    dist_px = line2_start[0] - line1_start[0]
    dist_mm = round(dist_px*mm_per_pixel, 4)

    cv2.line(img_color, line1_start, line1_end, (0, 255, 0), 2)
    cv2.line(img_color, line2_start, line2_end, (0, 255, 0), 2)

    # Add Road width value to image

    # text
    text = str(dist_mm) + ' mm'

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (line1_start[0]+50, round(img.shape[0]/2))

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    img_color = cv2.putText(img_color, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

    return [case_cond, img_color, edge_start, edge_end, num_edges, edge_count, dist_px, dist_mm]


def computeW_Rev(img): 
    
    avg_num_pixels = 159
    scaling_factor = 1.0
    mm_per_pixel = ((1/32)*25.4)/(scaling_factor*avg_num_pixels)
    edge_length_threshold = 55
    
    # Predefine arrays for data storage
    approx_edges = 5
    num_edges = np.zeros(img.shape[0]) #,dtype=np.uint16) 
    edge_start = np.zeros([img.shape[0],approx_edges])#,dtype=np.uint16)
    edge_end = np.zeros([img.shape[0],approx_edges])#,dtype=np.uint16)
    
    edge_count = 0
    k=0

    sse = False
    tse = False

    # start scanning from (0,0) until black pixel is found 
    # go across columns first

    for i in range(img.shape[0]):

        found_edge = False
        temp_edge_count = 0
        k=0    

        for j in range(img.shape[1]):

            if(img[i,j]<=50):
                # Black pixel found - edge
                if(found_edge==False):
                    found_edge = True
                    temp_edge_count += 1
                    num_edges[i] = temp_edge_count
                    edge_start[i][k] = j
                    k += 1

            else:
                if(found_edge):
                    edge_end[i][k-1] = j-1
                    found_edge = False      
        
    
    if max(num_edges)==2:

        # max num_edges = 2
        
        line1_start = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),0) 
        line1_end = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),img.shape[0])

        line2_start = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),0)
        line2_end = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),img.shape[0])

        edge_count = 2
        case_cond = 1
        
    elif max(num_edges)==3: 
        
        # max num_edges = 3                
        
        true2edge_start = edge_start[:,1][np.argwhere(num_edges==3)]
        true2edge_start = true2edge_start[true2edge_start<250]
        
        true3edge_start = edge_start[:,1][np.argwhere(num_edges==3)]
        true3edge_start = true3edge_start[np.logical_and(true3edge_start>250,true3edge_start<510)]
        
        true2edge_end = edge_end[:,1][np.argwhere(num_edges==3)][edge_end[:,1][np.argwhere(num_edges==3)]<250]
        
        true3edge_end = edge_end[:,1][np.argwhere(num_edges==3)][np.logical_and(edge_end[:,1][np.argwhere(num_edges==3)]>250,edge_end[:,1][np.argwhere(num_edges==3)]<510)]
        
        if (np.logical_and(true2edge_start.size != 0, len(true2edge_start)>=edge_length_threshold)):
            
            if (len(true2edge_start)>len(true2edge_end)):

                true2edge_start = np.array([true2edge_start[i] for i in range(len(true2edge_end))]) 

            elif (len(true2edge_start)<len(true2edge_end)):

                true2edge_end = np.array([true2edge_end[i] for i in range(len(true2edge_start))])            
            
            sse = True

        else:
            sse = False

        if (np.logical_and(true3edge_start.size != 0,len(true3edge_start)>=edge_length_threshold)):

            if (len(true3edge_start)>len(true3edge_end)):

                true3edge_start = np.array([true3edge_start[i] for i in range(len(true3edge_end))]) 

            elif (len(true3edge_start)<len(true3edge_end)):

                true3edge_end = np.array([true3edge_end[i] for i in range(len(true3edge_start))])  

            tse = True
            
        else:
            tse = False


        if (np.logical_and(sse==True,tse==False)):            
        
            line1_start = (round(np.mean((true2edge_start + true2edge_end)/2)),0) 
            line1_end = (round(np.mean((true2edge_start + true2edge_end)/2)),img.shape[0])

            if (np.logical_and(np.unique(num_edges)[0]==3,len(np.unique(num_edges)==1))):
                line2_start = (round(np.mean((edge_start[:,2][np.argwhere(num_edges==3)] + edge_end[:,2][np.argwhere(num_edges==3)])/2)),0)
                line2_end = (round(np.mean((edge_start[:,2][np.argwhere(num_edges==3)] + edge_end[:,2][np.argwhere(num_edges==3)])/2)),img.shape[0])

            else:
                line2_start = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),0)
                line2_end = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),img.shape[0])
                        
            edge_count = 3
            case_cond = 2

        elif (np.logical_and(sse==False,tse==True)):

            if (np.logical_and(np.unique(num_edges)[0]==3,len(np.unique(num_edges)==1))):
                line1_start = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==3)] + edge_end[:,0][np.argwhere(num_edges==3)])/2)),0)
                line1_end = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==3)] + edge_end[:,0][np.argwhere(num_edges==3)])/2)),img.shape[0])

            else:
                line1_start = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),0) 
                line1_end = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),img.shape[0])

            line2_start = (round(np.mean((true3edge_start + true3edge_end)/2)),0)
            line2_end = (round(np.mean((true3edge_start + true3edge_end)/2)),img.shape[0])
                        
            edge_count = 3
            case_cond = 3

        else:    
            
            # A third stray edge found beyond road 2nd edge to the right - noise 
            line1_start = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),0) 
            line1_end = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),img.shape[0])

            if (len(edge_start[:,1][np.argwhere(num_edges==2)]) > len(edge_start[:,1][np.argwhere(num_edges==3)])):

                line2_start = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),0)
                line2_end = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),img.shape[0])

            else:

                line2_start = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==3)] + edge_end[:,1][np.argwhere(num_edges==3)])/2)),0)
                line2_end = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==3)] + edge_end[:,1][np.argwhere(num_edges==3)])/2)),img.shape[0])
                            
            edge_count = 2
            case_cond = 4    
    
    elif max(num_edges)==4: 
        
        # max num_edges = 4
               
        true2edge_start = edge_start[:,1][np.argwhere(num_edges==3)]
        true2edge_start = true2edge_start[true2edge_start<250]
        true2edge_start = np.hstack((true2edge_start,edge_start[:,1][np.argwhere(num_edges==4)][edge_start[:,1][np.argwhere(num_edges==4)]<250]))

        true3edge_start = edge_start[:,1][np.argwhere(num_edges==3)]
        true3edge_start = true3edge_start[np.logical_and(true3edge_start>250,true3edge_start<500)]
        true3edge_start = np.hstack((true3edge_start,edge_start[:,2][np.argwhere(num_edges==4)][np.logical_and(edge_start[:,2][np.argwhere(num_edges==4)]>250,edge_start[:,2][np.argwhere(num_edges==4)]<500)]))

        true2edge_end = edge_end[:,1][np.argwhere(num_edges==3)][edge_end[:,1][np.argwhere(num_edges==3)]<250]
        true2edge_end = np.hstack((true2edge_end,edge_end[:,1][np.argwhere(num_edges==4)][edge_end[:,1][np.argwhere(num_edges==4)]<250]))

        true3edge_end = edge_end[:,1][np.argwhere(num_edges==3)][np.logical_and(edge_end[:,1][np.argwhere(num_edges==3)]>250,edge_end[:,1][np.argwhere(num_edges==3)]<510)]
        true3edge_end = np.hstack((true3edge_end,edge_end[:,2][np.argwhere(num_edges==4)][np.logical_and(edge_end[:,2][np.argwhere(num_edges==4)]>250,edge_end[:,2][np.argwhere(num_edges==4)]<510)]))

        if (np.logical_and(true2edge_start.size != 0, len(true2edge_start)>=edge_length_threshold)):
            
            if (len(true2edge_start)>len(true2edge_end)):

                true2edge_start = np.array([true2edge_start[i] for i in range(len(true2edge_end))]) 

            elif (len(true2edge_start)<len(true2edge_end)):

                true2edge_end = np.array([true2edge_end[i] for i in range(len(true2edge_start))])            
            
            sse = True

        else:
            sse = False

        if (np.logical_and(true3edge_start.size != 0,len(true3edge_start)>=edge_length_threshold)):

            if (len(true3edge_start)>len(true3edge_end)):

                true3edge_start = np.array([true3edge_start[i] for i in range(len(true3edge_end))]) 

            elif (len(true3edge_start)<len(true3edge_end)):

                true3edge_end = np.array([true3edge_end[i] for i in range(len(true3edge_start))])  

            tse = True
            
        else:
            tse = False


        if (np.logical_and(sse==True,tse==True)):

            # there is significant 2nd and 3rd edge             
                    
            if (np.logical_or(len(true2edge_start)!=len(true2edge_end),len(true3edge_start)!=len(true3edge_end))):
                print('true edge start end length not equal!')

            line1_start = (round(np.mean((true2edge_start + true2edge_end)/2)),0) 
            line1_end = (round(np.mean((true2edge_start + true2edge_end)/2)),img.shape[0])

            line2_start = (round(np.mean((true3edge_end + true3edge_end)/2)),0)
            line2_end = (round(np.mean((true3edge_end + true3edge_end)/2)),img.shape[0])
            
            edge_count = 4
            case_cond = 5

        elif (np.logical_and(sse==True,tse==False)):

            right_edge_start = edge_start[:,2][np.argwhere(num_edges==3)][edge_start[:,2][np.argwhere(num_edges==3)]>250]
            right_edge_start = np.hstack((right_edge_start,edge_start[:,1][np.argwhere(num_edges==2)][edge_start[:,1][np.argwhere(num_edges==2)]>250]))
            right_edge_end = edge_end[:,2][np.argwhere(num_edges==3)][edge_end[:,2][np.argwhere(num_edges==3)]>250]
            right_edge_end = np.hstack((right_edge_end,edge_end[:,1][np.argwhere(num_edges==2)][edge_end[:,1][np.argwhere(num_edges==2)]>250]))

            line1_start = (round(np.mean((true2edge_start + true2edge_end)/2)),0) 
            line1_end = (round(np.mean((true2edge_start + true2edge_end)/2)),img.shape[0])

            line2_start = (round(np.mean((right_edge_start + right_edge_end)/2)),0)
            line2_end = (round(np.mean((right_edge_start + right_edge_end)/2)),img.shape[0])
            
            edge_count = 3
            case_cond = 6

        elif (np.logical_and(sse==False,tse==True)):

            left_edge_start = edge_start[:,0][np.argwhere(num_edges==3)][np.logical_and(edge_start[:,0][np.argwhere(num_edges==3)]>135,edge_start[:,0][np.argwhere(num_edges==3)]<250)]
            left_edge_start = np.hstack((left_edge_start,edge_start[:,0][np.argwhere(num_edges==2)][np.logical_and(edge_start[:,0][np.argwhere(num_edges==2)]>135,edge_start[:,0][np.argwhere(num_edges==2)]<250)]))
            left_edge_end = edge_end[:,0][np.argwhere(num_edges==3)][np.logical_and(edge_end[:,0][np.argwhere(num_edges==3)]>135,edge_end[:,0][np.argwhere(num_edges==3)]<250)]
            left_edge_end = np.hstack((left_edge_end,edge_end[:,0][np.argwhere(num_edges==2)][np.logical_and(edge_end[:,0][np.argwhere(num_edges==2)]>135,edge_end[:,0][np.argwhere(num_edges==2)]<250)]))

            line1_start = (round(np.mean((left_edge_start + left_edge_end)/2)),0) 
            line1_end = (round(np.mean((left_edge_start + left_edge_end)/2)),img.shape[0])

            line2_start = (round(np.mean((true3edge_end + true3edge_end)/2)),0)
            line2_end = (round(np.mean((true3edge_end + true3edge_end)/2)),img.shape[0])
            
            edge_count = 3
            case_cond = 7

        else:               
                
            line1_start = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),0) 
            line1_end = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),img.shape[0])

            line2_start = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),0)
            line2_end = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),img.shape[0])
                
            edge_count = 2
            case_cond = 8
            
    else:

        # greater than 4 max edges case is typically noisy non-edges
        
        line1_start = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),0) 
        line1_end = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),img.shape[0])

        line2_start = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),0)
        line2_end = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),img.shape[0])
        
        edge_count = 2
        case_cond = 9
 
        
    # convert to BGR image and draw line
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # computed road width in pixels 
    dist_px = line2_start[0] - line1_start[0]
    dist_mm = round(dist_px*mm_per_pixel,4)
    
    cv2.line(img_color,line1_start,line1_end,(0,255,0),2)
    cv2.line(img_color,line2_start,line2_end,(0,255,0),2)
    
    # Add Road width value to image
    
    # text
    text = str(dist_mm) + ' mm'
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # org
    org = (line1_start[0]+50, round(img.shape[0]/2))
  
    # fontScale
    fontScale = 1
   
    # Blue color in BGR
    color = (255, 0, 0)
  
    # Line thickness of 2 px
    thickness = 2
   
    # Using cv2.putText() method
    img_color = cv2.putText(img_color, text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)      
    
    return [case_cond,img_color,edge_start,edge_end,num_edges,edge_count,dist_px,dist_mm]



################ MAIN ################

# Setup IO
GPIO.setmode(GPIO.BCM)
GPIO.setup(26,GPIO.IN) # set pin 26 as digital IN from Robot for camera start trigger

vidCount = 0
start_timestamp = False
end_timestamp = False
acq_trigger_started = False
acq_trigger_ended = False
acq_trigger_wait = False

columns = ['Video_Count', 'Layer', 'Speed', 'Start_Timestamp', 'End_Timestamp', 'Time_Diff']
lst = []
vfc = 1
vR = 10
layer = 3

try:
        
    cap = cv2.VideoCapture("/dev/video1")

    # check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error initializing camera stream, check camera connections. Exiting ...")
        exit(0)
    
    while (cap.isOpened()):

        if not acq_trigger_wait:
            print("Waiting for start video acqusition trigger . . . ")
            acq_trigger_wait = True

        while(GPIO.input(26)==0):
       
            if not acq_trigger_started:

                
                #fps = cap.get(cv2.CAP_PROP_FPS)
                #print('Video frame rate={0}'.format(fps))

                #pframe_width = 645
                #pframe_height = 345

                #psize = (pframe_width, pframe_height)

                #vidName = input("Enter Video Name: ")
                #result = cv2.VideoWriter(
                #    vidName + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 30, psize)

                start_time = datetime.datetime.now()

                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))

                size = (frame_width, frame_height)                
                
                vidCount = vidCount + 1
                vidName = "vid" + str(vidCount)
                result = cv2.VideoWriter(str(vidName) + ".avi",cv2.VideoWriter_fourcc(*'XVID'),30, size)

                print('The video acqusition for vid{0} has started at {1}'.format(vidCount,start_time))
                acq_trigger_started = True
                acq_trigger_wait = False
                acq_trigger_ended = False
                                

            frame_exists, frame = cap.read()
            #print('frame count: {0}'.format(frameCount))
            #print('ret: {0}'.format(ret))

            if frame_exists:

                    # Write the frame into the
                    # file 'filename.avi'
                    result.write(frame)

                    # show frame
                    cv2.imshow('Live Video', frame)

                    # # to calculate the transformation matrix
                    # pt1 = [192.30, 343.00]  # (x,y) - Top left
                    # pt2 = [1079.0, 379.80]  # (x,y) - Top right
                    # pt3 = [153.50, 571.90]  # (x,y) - bottom left
                    # pt4 = [1107.10, 611.70]  # (x,y) - bottom Right

                    # # Actual Dimensions of region selected by 4 points
                    # # mm Actual ruler width measured using calipers
                    # scale_width = round(11.7348*200)
                    # # mm Height based on selected 4 points for perspective transform
                    # scale_height = round(6.35*200)

                    # # pt1 = [137,248]  # (x,y) - Top left
                    # # pt2 = [1001,283]  # (x,y) - Top right
                    # # pt3 = [60,648]  # (x,y) - bottom left
                    # # pt4 = [1041,688] # (x,y) - bottom Right

                    # # Actual Dimensions of region selected by 4 points
                    # # scale_width = round((14/32)*25.40*200) # mm Actual ruler width measured using calipers
                    # # scale_height = round(6.35*200)   # mm Height based on selected 4 points for perspective transform

                    # # call function - correct perspective transform
                    # [img_rgb, imgPersp, imgGrayPersp] = perspCorrection(
                    #     frame, pt1, pt2, pt3, pt4, scale_width, scale_height)

                    # # Filter grayscale image
                    # # Bilateral filter
                    # bfblur = bilateralFilter(imgGrayPersp)

                    # # Extract top - bottom smaller regions
                    # tStart = [655, 0]
                    # tEnd = [1300, 345]
                    # bStart = [655, 925]
                    # bEnd = [1300, 1270]

                    # [img_top, img_bottom] = extractTopBottom(
                    #     bfblur, tStart, tEnd, bStart, bEnd)

                    # # convert to RGB image and draw line
                    # img_ROI = cv2.cvtColor(imgGrayPersp, cv2.COLOR_GRAY2RGB)
                    # img_ROI = cv2.rectangle(
                    #     img_ROI, (bStart[0], bStart[1]), (bEnd[0], bEnd[1]), (255, 0, 0), 5)
                    # #img_ROI = cv2.rectangle(img_ROI, (tStart[0],tStart[1]), (tEnd[0],tEnd[1]), (0,0,255), 5)

                    # #dstPathTop = 'Perspective Corrected\\Top\\'
                    # # cv2.imwrite(dstPathTop+'top'+str(i+1)+'.jpg',img_top)

                    # #dstPathBtm = 'Perspective Corrected\\Bottom\\'
                    # # cv2.imwrite(dstPathBtm+'bottom'+str(i+1)+'.jpg',img_bottom)

                    # # Thresholding - Adaptive Gaussian
                    # #thresh_top = gAdaptiveThresholding(img_top)
                    # thresh_bottom = gAdaptiveThresholding(img_bottom)

                    # #dstPathTop = 'Perspective Corrected\\Top\\'
                    # # cv2.imwrite(dstPathTop+'top'+str(i+1)+'.jpg',img_top)

                    # #dstPathThBtm = 'Filtered Images\\Bilateral Blur\\Bottom\\Thresholding AG\\'
                    # # cv2.imwrite(dstPathThBtm+'threshAG_bottom'+str(i+1)+'.jpg',thresh_bottom)

                    # # create kernel
                    # kernel1 = np.ones((8, 2), np.uint8)
                    # kernel2 = np.ones((5, 2), np.uint8)

                    # # perform morph operations
                    # # binImgTop=morphOps(thresh_top,kernel1,kernel2)
                    # binImgBtm = morphOps(thresh_bottom, kernel1, kernel2)

                    # # save images
                    # #dstPathMBtm = 'Filtered Images\\Bilateral Blur\\Bottom\\Morph Ops\\'
                    # # cv2.imwrite(dstPathMBtm+'binary_bottom'+str(i+1)+'.jpg',binImgBtm)

                    # # Extrusion width measurement
                    # #[top_img_color,top_edge_start,top_edge_end,top_num_edges,top_edge_count,top_edge_dist_pixels,top_edge_dist] = computeW(binImgTop)
                    # [bottom_case_cond, bottom_img_color, bottom_edge_start, bottom_edge_end, bottom_num_edges,
                    #     bottom_edge_count, bottom_edge_dist_pixels, bottom_edge_dist] = computeW_Rev(binImgBtm)

                    # # Write the frame into the file 'filename.avi'
                    # result.write(bottom_img_color)

                    # # show frame
                    # cv2.imshow('Road Measurement', bottom_img_color)

                    # #dstPathRMBtm = 'Road Measurements\\Bottom\\Revised ComputeW\\'
                    # # cv2.imwrite(dstPathRMBtm+'rm_bottom'+str(i+1)+'.jpg',bottom_img_color)

                    # # Store results in dataframe
                    # # lst.append([i+1,bottom_case_cond,bottom_edge_count,bottom_edge_dist])
                    # # lst.append([i+1,top_edge_count,bottom_edge_count,top_edge_dist,bottom_edge_dist])

                    # #print('Finished processing frame {0}'.format(i+1))

                    # # save summary figure - Bottom
                    # #summPath = 'Filtered Images\\Bilateral Blur\\Bottom\\Summary - Original Method\\'
                    # #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # # fig = plt.figure() # Remove memory leaks when plotting multiple figures inside a loop
                    # #titles = ['Bilateral Blur', 'AG Thresholding', 'Morph Ops', 'Edge Detection+Measurement']
                    # #images = [img, thresh_bottom, binImgBtm, bottom_img_color]

                    # # for x in range(4):
                    # #    plt.subplot(2,2,x+1),plt.imshow(images[x],'gray')
                    # #    plt.title(titles[x],fontsize=10,wrap=True)
                    # #    plt.xticks([]),plt.yticks([])

                    # #plt.suptitle('Summary Bottom - Image Processing for Frame-{0}'.format(i+1),fontsize=15,wrap=True)
                    # # plt.savefig(summPath+'filter2End_OrigMethod'+str(i+1)+'.png')
                    # # plt.close(fig) # close figure to conserve memory. If not close to 3 GB of RAM usage!!!

                    # Press S on keyboard to stop the process
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            else:
                print("Can't retrieve frame - stream may have ended. Exiting..")
                break
        
        if not acq_trigger_ended:
            if vidCount >= 1:
                end_time = datetime.datetime.now()
                print('The video acqusition for vid{0} has ended at {1}'.format(vidCount,end_time))
                acq_trigger_ended = True
                acq_trigger_started = False 
                
                # Closes all the frames
                result.release()
                cv2.destroyAllWindows()
                print("The video was successfully saved")
                
                time_diff = (end_time - start_time)
                execution_time = time_diff.total_seconds() * 1000
                lst.append([vidCount,layer,vR,start_time,end_time,time_diff])
                
                # Update layer count and speed
                vR = vR+10
                
                if(vR > 50):
                    vR = 10
                    layer = layer + 1

except KeyboardInterrupt:
    # user pressed ctrl + C
    print("Program terminated by user. Exiting gracefully . . . ")
    if(cap.isOpened()):
        cap.release()
        result.release()
    GPIO.cleanup()
    cv2.destroyAllWindows()

    # Save pandas dataframe to excel/csv
    video_timestamps = pd.DataFrame(lst,columns=columns)
    video_timestamps.to_excel('Video_Timestamps_' + vfc + '.xlsx')
    exit(0)

