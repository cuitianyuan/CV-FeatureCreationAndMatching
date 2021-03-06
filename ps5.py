"""Problem Set 5: Harris, ORB, RANSAC."""

import numpy as np
import cv2
import math
import random

def gradient_x(image):
    """Computes the image gradient in X direction.

    This method returns an image gradient considering the X direction. See cv2.Sobel.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in X direction with values in [-1.0, 1.0].
    """
   
    image = cv2.GaussianBlur(image,(5,5),0) 
    #laplacian = cv2.Laplacian(image,cv2.CV_64F)
    gx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=-1)
    
    return gx


def gradient_y(image):
    """Computes the image gradient in Y direction.

    This method returns an image gradient considering the Y direction. See cv2.Sobel.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in Y direction with values in [-1.0, 1.0].
    """
    image = cv2.GaussianBlur(image,(5,5),0)
    gy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=-1)
    return gy

def make_image_pair(image1, image2):
    """Adjoins two images side-by-side to make a single new image.

    The output dimensions must take the maximum height from both images for the total height.
    The total width is found by adding the widths of image1 and image2.

    Args:
        image1 (numpy.array): first image, could be grayscale or color (BGR).
                              This array takes the left side of the output image.
        image2 (numpy.array): second image, could be grayscale or color (BGR).
                              This array takes the right side of the output image.

    Returns:
        numpy.array: combination of both images, side-by-side, same type as the input size.
    """
    image12 = np.concatenate((image1, image2), axis=1)
    return image12

def harris_response(ix, iy, kernel_dims, alpha):
    """Computes the Harris response map using given image gradients.

    Args:
        ix (numpy.array): image gradient in the X direction with values in [-1.0, 1.0].
        iy (numpy.array): image gradient in the Y direction with the same shape and type as Ix.
        kernel_dims (tuple): 2D windowing kernel dimensions. ie. (3, 3)  (3, 5).
        alpha (float): Harris detector parameter multiplied with the square of trace.

    Returns:
        numpy.array: Harris response map, same size as inputs, floating-point.
    """
    ######## start
    
    Ixx=ix*ix
    Ixy=iy*ix
    Iyy=iy*iy
    
    Ixx_w=cv2.GaussianBlur(Ixx,kernel_dims,0.0)
    Ixy_w=cv2.GaussianBlur(Ixy,kernel_dims,0.0)
    Iyy_w=cv2.GaussianBlur(Iyy,kernel_dims,0.0)
    
    harris = Ixx_w * Iyy_w - np.square(Ixy_w) - alpha * (Ixx_w + Iyy_w) * (Ixx_w + Iyy_w)
    harrisNorm = cv2.normalize(harris, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #import matplotlib.pyplot as plt
    #plt.imshow(harrisNorm, "gray")
    return harrisNorm
 

def find_corners(r_map, threshold, radius):
    """Finds corners in a given response map.

    This method uses a circular region to define the non-maxima suppression area. For example,
    let c1 be a corner representing a peak in the Harris response map, any corners in the area
    determined by the circle of radius 'radius' centered in c1 should not be returned in the
    peaks array.

    Make sure you account for duplicate and overlapping points.

    Args:
        r_map (numpy.array): floating-point response map, e.g. output from the Harris detector.
        threshold (float): value between 0.0 and 1.0. Response values less than this should
                           not be considered plausible corners.
        radius (int): radius of circular region for non-maximal suppression.

    Returns:
        numpy.array: peaks found in response map R, each row must be defined as [x, y]. Array
                     size must be N x 2, where N are the number of points found.
    """
    r_map1 = np.copy(r_map)
    r_map1[ r_map1 < threshold ] = 0 

    data_max = np.zeros(r_map1.shape)
    ind = np.nonzero(r_map1)
    for n in range(len(ind[0])):
        i = ind[0][n]
        j = ind[1][n]
        frame = r_map1[max((i-radius), 0):min((i+radius),r_map1.shape[0]),max((j-radius),0):min((j+radius),r_map1.shape[1])]
        if r_map1 [i,j]< np.max(frame) : # not local max
            data_max [i,j]  = 0
        elif np.max(data_max[max((i-radius), 0):min((i+radius),data_max.shape[0]),max((j-radius),0):min((j+radius),data_max.shape[1])])>0: 
            # Tie, and already as a max t
            data_max [i,j]  = 0
        else: 
            data_max [i,j] = r_map1[i, j]

    col_ind, row_ind = np.nonzero(data_max) 
    corners = []
    for i in range(len(row_ind)):
        corners.append([row_ind[i], col_ind[i]])
    return np.array(corners)

def draw_corners(image, corners):
    """Draws corners on (a copy of) the given image.

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        corners (numpy.array): peaks found in response map R, as a sequence of [x, y] coordinates.
                               Array size must be N x 2, where N are the number of points found.
    Returns:
        numpy.array: copy of the input image with corners drawn on it, in color (BGR).
    """
    image_RGB = cv2.cvtColor(image.astype('float32'),cv2.COLOR_GRAY2RGB)
    
    for i in range(len(corners)):
      x, y = corners[i]
      cv2.circle(img = image_RGB, center = (x, y), radius = 3, color=(0,0,1.) )
    
    return image_RGB  #image_RGB*255.
# -- auto graded
def gradient_angle(ix, iy):
    """Computes the e angle (orientation) image given the X and Y gradients.

    Args:
        ix (numpy.array): image gradient in X direction.
        iy (numpy.array): image gradient in Y direction, same size and type as Ix

    Returns:
        numpy.array: gradient angle image, same shape as ix and iy. Values must be in degrees [0.0, 360).
    """
    Gdir = ix
    for i in range(len(ix)):
        for j in range (len(ix[0])):
            Gdir[i][j] = ( math.atan2( iy[i][j], ix[i][j])*180/math.pi  + 360.) % 360.
    return np.array(Gdir)

# -- auto graded
def get_keypoints(points, angle, size, octave=0):
    """Creates OpenCV KeyPoint objects given interest points, response map, and angle images.

    See cv2.KeyPoint and cv2.drawKeypoint.

    Args:
        points (numpy.array): interest points (e.g. corners), array of [x, y] coordinates.
        angle (numpy.array): gradient angle (orientation) image, each value in degrees [0, 360).
                             Keep in mind this is a [row, col] array. To obtain the correct
                             angle value you should use angle[y, x].
        size (float): fixed _size parameter to pass to cv2.KeyPoint() for all points.
        octave (int): fixed _octave parameter to pass to cv2.KeyPoint() for all points.
                      This parameter can be left as 0.

    Returns:
        keypoints (list): a sequence of cv2.KeyPoint objects
    """ 
    keypoints = []
    for i in range(len(points)):
        keypoint = cv2.KeyPoint(x=points[i][0], y=points[i][1]
                                , _size=size
                                , _angle=  angle[points[i][1], points[i][0]]       
                                , _octave=octave)
        keypoints.append(keypoint)
    return keypoints
    
    
def get_descriptors(image, keypoints):
    """Extracts feature descriptors from the image at each keypoint.

    This function finds descriptors following the methods used in cv2.ORB. You are allowed to
    use such function or write your own.

    Args:
        image (numpy.array): input image where the descriptors will be computed from.
        keypoints (list): a sequence of cv2.KeyPoint objects.

    Returns:
        tuple: 2-element tuple containing:
            descriptors (numpy.array): 2D array of shape (len(keypoints), 32).
            new_kp (list): keypoints from ORB.compute().
    """
    # Initiate ORB detector
    orb = cv2.ORB() 
    #normalized version of input image
    img_norm = cv2.normalize(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
    new_kp, descriptors = orb.compute(img_norm, keypoints)
    return descriptors,  new_kp 


# -- auto graded
def match_descriptors(desc1, desc2):
    """Matches feature descriptors obtained from two images.

    Use cv2.NORM_HAMMING and cross check for cv2.BFMatcher. Return the matches sorted by distance.

    Args:
        desc1 (numpy.array): descriptors from image 1, as returned by ORB.compute().
        desc2 (numpy.array): descriptors from image 2, same format as desc1.

    Returns:
        list: a sequence (list) of cv2.DMatch objects containing corresponding descriptor indices.
    """

    # Note: You can use OpenCV's descriptor matchers, or write your own!
    #       Make sure you use Hamming Normalization to match the autograder.
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(desc1,desc2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    return matches


def draw_matches(image1, image2, kp1, kp2, matches):
    """Shows matches by drawing lines connecting corresponding keypoints.

    Results must be presented joining the input images side by side (use make_image_pair()).

    OpenCV's match drawing function(s) are not allowed.

    Args:
        image1 (numpy.array): first image
        image2 (numpy.array): second image, same type as first
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2
        matches (list): list of matching keypoint index pairs (as cv2.DMatch objects)

    Returns:
        numpy.array: image1 and image2 joined side-by-side with matching lines;
                     color image (BGR), uint8, values in [0, 255].
    """
    img1 = cv2.normalize(image1*255., alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
    img2 = cv2.normalize(image2*255., alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    rows1 = image1.shape[0]
    cols1 = image1.shape[1]
    rows2 = np.asarray(image2).shape[0]
    cols2 = np.asarray(image2).shape[1]
    out  = make_image_pair(img1, img2 )
    out  = cv2.cvtColor(out ,cv2.COLOR_GRAY2RGB)
    cl = []
    for i in range(1000):
        cl.append((255*np.random.rand(), 255*np.random.rand(), 255*np.random.rand()))
    
    for k in range(len(matches)) :

        img1_idx = matches[k].queryIdx
        img2_idx = matches[k].trainIdx

        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1),int(y1)), 4, cl[k], 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, cl[k], 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), cl[k], 1 )

    return out



def compute_translation_RANSAC(kp1, kp2, matches, thresh):
    """Computes the best translation vector using RANSAC given keypoint matches.

    Args:
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1.
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2.
        matches (list): list of matches (as cv2.DMatch objects).
        thresh (float): offset tolerance in pixels which decides if a match forms part of
                        the consensus. This value can be seen as a minimum delta allowed
                        between point components.

    Returns:
        tuple: 2-element tuple containing:
            translation (numpy.array): translation/offset vector <x, y>, array of shape (2, 1).
            good_matches (list): consensus set of matches that agree with this translation.
    """

    best_tt_in = 0
    best_inlier_ind = []
    best_dx = 0.
    best_dy = 0.
    max_iter= 100
    for total_iter in range(max_iter):
        s = random.sample(range(len(matches)/2), 2)
        inlier_ind = s  #starting point is always a inlier
        
        tt_in = 0
        
        (x1a,y1a) = kp1[matches[s[0]].queryIdx].pt
        (x2a,y2a) = kp2[matches[s[0]].trainIdx].pt
        (x1b,y1b) = kp1[matches[s[1]].queryIdx].pt
        (x2b,y2b) = kp2[matches[s[1]].trainIdx].pt
        (dx, dy)  =  (.5*(x2a-x1a) + .5*(x2b-x1b), .5*(y2a-y1a) + .5*(y2b-y1b))
        dx_l = [x2a-x1a, x2b-x1b]
        dy_l = [y2a-y1a, y2b-y1b]
    
        for i in [x for x in xrange(len(matches)) if x not in s]:
            (x1,y1) = kp1[matches[i].queryIdx].pt
            (x2,y2) = kp2[matches[i].trainIdx].pt
            print x2-x1, y2-y1
            if ((x1+dx-thresh) <= x2 <= (x1+dx+thresh)) & ((y1+dy-thresh) <= y2 <= (y1+dy+thresh)):
                tt_in += 1
                inlier_ind.append(i)
                dx_l.append(x2-x1)
                dy_l.append(y2-y1)
        
        if tt_in > best_tt_in:
            best_tt_in = tt_in
            best_inlier_ind = inlier_ind
            best_dx = np.mean(dx_l)
            best_dy = np.mean(dy_l)
        print tt_in, best_tt_in,tt_in > best_tt_in, best_inlier_ind
    #print tt_in, best_tt_in,dx_l,dy_l

    return np.asarray([best_dx,best_dy]).reshape((2,1)) ,[matches[i] for i in best_inlier_ind]  


def compute_similarity_RANSAC(kp1, kp2, matches, thresh):
    """Computes the best similarity transform using RANSAC given keypoint matches.

    Args:
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1.
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2.
        matches (list): list of matches (as cv2.DMatch objects).
        thresh (float): offset tolerance in pixels which decides if a match forms part of
                        the consensus. This value can be seen as a minimum delta allowed
                        between point components.

    Returns:
        tuple: 2-element tuple containing:
            m (numpy.array): similarity transform matrix of shape (2, 3).
            good_matches (list): consensus set of matches that agree with this transformation.
    """

    # Note: this function must use the RANSAC method. If you implement any non-RANSAC approach
    # (i.e. brute-force) you will not get credit for either the autograder tests or the report
    # sections that depend of this function.
    
    # simple average
    best_tt_in = 0
    best_inlier_ind = []
    best_dx = 0.
    best_dy = 0.
    max_iter= 100
    for total_iter in range(max_iter):
        s = random.sample(range(len(matches)), 2)\
        inlier_ind = s  #starting point is always a inlier
        
        tt_in = 0\

        (p1x,p1y) = kp1[matches[s[0]].queryIdx].pt
        (p2x,p2y) = kp1[matches[s[1]].queryIdx].pt 
        (p1x_,p1y_) = kp2[matches[s[0]].trainIdx].pt
        (p2x_,p2y_) = kp2[matches[s[1]].trainIdx].pt 
        
        A = np.asarray([[p1x,-p1y,1,0]
                        ,[p1y,p1x,0,1]
                        ,[p2x,-p2y,1,0]
                        ,[p2y,p2x,0,1]])
        
        B =np.asarray([p1x_, p1y_, p2x_, p2y_]).T
                     
                     
        sim =  np.linalg.lstsq(A , B ) [0]
               
        sim_mat = np.asarray( [[sim[0], -sim[1], sim[2]]
                              ,[sim[1],  sim[0], sim[3]]])
        
        for i in [x for x in xrange(len(matches)) if x not in s]:
            # [x1, y1, 1]
            left = np.asarray([kp1[matches[i].queryIdx].pt[0], kp1[matches[i].queryIdx].pt[1] ,1.]).T
            # predictive [u', v']
            (u_pred, v_pred) = np.dot(sim_mat , left)
            
            # [x2, y2] in the right image
            (x2,y2) = kp2[matches[i].trainIdx].pt   
            print ( x2- int(u_pred) , (y2-int(v_pred)) ) , x2, int(u_pred),y2, int(v_pred)
            if  ( abs(x2-int(u_pred)) <= thresh )  & (  abs(y2-int(v_pred)) <= thresh):
                tt_in += 1
                inlier_ind.append(i)
        
        if tt_in > best_tt_in:
            best_tt_in = tt_in
            best_inlier_ind = inlier_ind
            best_sim_mat = sim_mat
        print tt_in, best_tt_in,tt_in > best_tt_in, best_inlier_ind


    return best_sim_mat ,[matches[i] for i in best_inlier_ind]  


def compute_affine_RANSAC(kp1, kp2, matches, thresh):
    """ Compute the best affine transform using RANSAC given keypoint matches.

    Args:
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2
        matches (list): list of matches (as cv2.DMatch objects)
        thresh (float): offset tolerance in pixels which decides if a match forms part of
                        the consensus. This value can be seen as a minimum delta allowed
                        between point components.

    Returns:
        tuple: 2-element tuple containing:
            m (numpy.array): affine transform matrix of shape (2, 3)
            good_matches (list): consensus set of matches that agree with this transformation.
    """

    # Note: this function must use the RANSAC method. If you implement any non-RANSAC approach
    # (i.e. brute-force) you will not get credit for either the autograder tests or the report
    # sections that depend of this function.
    
    # simple average
    best_tt_in = 0
    best_inlier_ind = []
    best_dx = 0.
    best_dy = 0.
    max_iter= 100
    
    for total_iter in range(max_iter):
        s = random.sample(range(len(matches)  ), 3)
        inlier_ind = s  #starting point is always a inlier
        
        tt_in = 0
        
        u = [kp1[matches[s[0]].queryIdx].pt[0], kp1[matches[s[1]].queryIdx].pt[0], kp1[matches[s[2]].queryIdx].pt[0]]
        v = [kp1[matches[s[0]].queryIdx].pt[1], kp1[matches[s[1]].queryIdx].pt[1], kp1[matches[s[2]].queryIdx].pt[1]]
        
        x = [kp2[matches[s[0]].trainIdx].pt[0], kp2[matches[s[1]].trainIdx].pt[0], kp2[matches[s[2]].trainIdx].pt[0]]
        y = [kp2[matches[s[0]].trainIdx].pt[1], kp2[matches[s[1]].trainIdx].pt[1], kp2[matches[s[2]].trainIdx].pt[1]]
        
        
        B = np.asarray([x[0], y[0], x[1], y[1], x[2], y[2]] ).T
        
        A = np.asarray([     [u[0] , v[0] , 1  , 0    , 0   , 0]
                            ,[0    ,0     , 0  , u[0] ,v[0] , 1]
                            ,[u[1] ,v[1]  , 1  , 0    ,0    , 0]
                            ,[0    ,0     , 0  , u[1] ,v[1] , 1]
                            ,[u[2] , v[2] , 1  , 0    ,0    , 0]
                            ,[0    ,0     , 0  , u[2] , v[2], 1]])
        
        aff = np.linalg.lstsq(A , B ) [0]
        aff_mat = np.asarray( [[aff[0],  aff[1], aff[2]]
                              ,[aff[3],  aff[4], aff[5]]])
        
        for i in [x for x in xrange(len(matches)) if x not in s]:
            # [x1, y1, 1]
            left = np.asarray([kp1[matches[i].queryIdx].pt[0], kp1[matches[i].queryIdx].pt[1] ,1.]).T
            # predictive [u', v']
            (u_pred, v_pred) = np.dot(aff_mat , left)
            # [x2, y2] in the right image
            (x2,y2) = kp2[matches[i].trainIdx].pt
            if  ( abs(x2-int(u_pred)) <= thresh )  & (  abs(y2-int(v_pred)) <= thresh):
                tt_in += 1
                inlier_ind.append(i)
        
        if tt_in > best_tt_in:
            best_tt_in = tt_in
            best_inlier_ind = inlier_ind
            best_aff_mat = aff_mat
        print tt_in, best_tt_in,tt_in > best_tt_in, best_inlier_ind

    return best_aff_mat ,[matches[i] for i in best_inlier_ind]  


def warp_img(img_a, img_b, m):
    """Warps image B using a transformation matrix.

    Keep in mind:
    - Write your own warping function. No OpenCV functions are allowed.
    - If you see several black pixels (dots) in your image, it means you are not
      implementing backwards warping.
    - If line segments do not seem straight you can apply interpolation methods.
      https://en.wikipedia.org/wiki/Interpolation
      https://en.wikipedia.org/wiki/Bilinear_interpolation

    Args:
        img_a (numpy.array): reference image.
        img_b (numpy.array): image to be warped.
        m (numpy.array): transformation matrix, array of shape (2, 3).

    Returns:
        tuple: 2-element tuple containing:
            warpedB (numpy.array): warped image.
            overlay (numpy.array): reference and warped image overlaid. Copy the reference
                                   image in the red channel and the warped image in the
                                   green channel
    """



    warpedB = np.zeros(img_a.shape)
    (bx, by) = img_b.shape         
    (ax, ay) = img_a.shape     
    for j in range(bx):
        for k in range(by):
            target =   np.dot( m, np.asarray([[j, k, 1]]).T  )  \
            if  ( (int(target[0]) >= 0 ) &  (int(target[0]) < ax ) & (int(target[1]) >= 0 ) & ( int(target[1]) < ay)): #find a map within the rage of img_a range
                warpedB[int(target[0]), int(target[1])] = img_b[j, k] 


    overlay = np.zeros((ax, ay,3))
    overlay[:,:,1] = warpedB  #green
    overlay[:,:,2] = img_a    #red 
    
    return  (warpedB*255.).astype(np.uint8), (overlay*255.).astype(np.uint8)
     
