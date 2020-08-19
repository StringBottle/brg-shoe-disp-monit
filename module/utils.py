#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This module uses Numpy & OpenCV(opencv-python). Please install before use it.

import os.path
import numpy as np
import cv2

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        imageBGR = cv2.imdecode(n, flags)
        return cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(e)
        return None

def normalizeControlPoints(pts):
    
    """
    This code is a Python implementation of Matlab code
    
    'images.geotrans.internal.normalizeControlPoints'
    
    Reference : 
    
    [1] Matlab Code Download link : https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/46441/versions/3/previews/MarsRover2014/matlab/calibration/myNormalizeControlPoints.m/index.html
    
    [2] Hartley, R; Zisserman, A. "Multiple View Geometry in Computer Vision." Cambridge University Press, 2003. pg. 180-181.

    """
    
    # Define N, the number of control points
    N = pts.shape[0]

    # Compute [xCentroid,yCentroid]
    cent = np.mean(pts, 0)

    # Shift centroid of the input points to the origin.
    ptsNorm = np.zeros((4, 2))
    ptsNorm[:, 0] = pts[:, 0] - cent[0]
    ptsNorm[:, 1] = pts[:, 1] - cent[1]

    sumOfPointDistancesFromOriginSquared = np.sum(np.power(np.hypot(ptsNorm[:, 0], ptsNorm[:, 1]), 2))

    if sumOfPointDistancesFromOriginSquared > 0:
        scaleFactor = np.sqrt(2 * N) / np.sqrt(sumOfPointDistancesFromOriginSquared)
        
    #     else:
    # % If all input control points are at the same location, the denominator
    # % of the scale factor goes to 0. Don't rescale in this case.
    #         if isa(pts,'single'):
    #             scaleFactor = single(1);
    #         else:
    #             scaleFactor = 1.0;
    # % Scale control points by a common scalar scale factor
    
    ptsNorm = np.multiply(ptsNorm, scaleFactor)

    normMatrixInv = np.array([
        [1 / scaleFactor, 0, 0],
        [0, 1 / scaleFactor, 0],
        [cent[0], cent[1], 1]])

    return ptsNorm, normMatrixInv


def findProjectiveTransform(uv, xy):
    
    
    """
    This code is a Python implementation of Matlab code
    
    'findProjectiveTransform' in 'fitgeotrans' 
    
    Reference : to-be added here
    
    """
        
    uv, normMatrix1 = normalizeControlPoints(uv);
    xy, normMatrix2 = normalizeControlPoints(xy);

    minRequiredNonCollinearPairs = 4;
    M = xy.shape[0]
    x = xy[:, 0][:, np.newaxis]
    y = xy[:, 1][:, np.newaxis]
    vec_1 = np.ones((M, 1))
    vec_0 = np.zeros((M, 1))
    u = uv[:, 0][:, np.newaxis]
    v = uv[:, 1][:, np.newaxis]

    U = np.vstack([u, v])
    X = np.hstack([[x, y, vec_1, vec_0, vec_0, vec_0, np.multiply(-u, x), np.multiply(-u, y)],
                   [vec_0, vec_0, vec_0, x, y, vec_1, np.multiply(-v, x), np.multiply(-v, y)]]).squeeze().T

    # We know that X * Tvec = U
    if np.linalg.matrix_rank(X) >= 2 * minRequiredNonCollinearPairs:
        Tvec = np.linalg.lstsq(X, U, rcond=-1)[0]
    #     else :
    #         error(message('images:geotrans:requiredNonCollinearPoints', minRequiredNonCollinearPairs, 'projective'))

    #      We assumed I = 1;
    Tvec = np.append(Tvec, 1)
    Tinv = np.reshape(Tvec, (3, 3));

    Tinv = np.linalg.lstsq(normMatrix2, np.matmul(Tinv, normMatrix1), rcond=-1)[0]
    T = np.linalg.inv(Tinv)
    T = T / T[2, 2]

    return T
