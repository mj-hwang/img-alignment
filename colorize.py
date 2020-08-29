# CS294-26: Project code
# Author: Minjune Hwang

import numpy as np
import skimage as sk
import skimage.io as skio

def generate_output(imname, pyramid=False):
    print(imname)

    # read in the image
    im = skio.imread(os.path.join("data", imname))  

    # convert to double (might want to do this later on to save memory) 
    im = sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    im_unaligned = np.dstack([r, g, b])
    f_unaligned = os.path.splitext(imname)[0] +"_unaligned.jpg"
    skio.imsave(os.path.join("result", f_unaligned), im_unaligned)
    
    # align the images
    print("aligning green")
    if pyramid:
        ag = align_pyramid(g, b)
        # ag = align_edges_pyramid(g, b)
    else:
        ag = align(g, b)
    
    print("aligning red")
    if pyramid:
        ar = align_pyramid(r, b)
        # ar = align_edges_pyramid(r, b)
    else:
        ar = align(r, b)
    
    im_aligned = np.dstack([ar, ag, b])
    f_aligned = os.path.splitext(imname)[0] +"_aligned.jpg"
    skio.imsave(os.path.join("result", f_aligned), im_aligned)
    print("-------------------------")

def align(im1, im2):
    best_x = 0
    best_y = 0
    best_dist = np.inf
    
    h = im2.shape[0]
    w = im2.shape[1]
    
    for i in range(-15, 15):
        for j in range(-15, 15):
            # shift image
            shifted_im1 = np.roll(np.roll(im1, i, axis=0), j, axis=1)

            # calculate SSD of the central part
            # to avoid the edge affecting the distance
            diff = (shifted_im1-im2)[h//4:3*h//4, w//4:3*w//4]
            dist = np.sum(diff**2)

            if dist < best_dist:
                best_x = i
                best_y = j
                best_dist = dist

    dx = best_x
    dy = best_y

    # print offsets (which are total dx and dy)
    print("total x-axis offset of im1 is {}".format(dx))
    print("total y-axis offset of im1 is {}".format(dy)) 
    
    return np.roll(np.roll(im1, dx, axis=0), dy, axis=1)

def align_pyramid(im1, im2):
    # save total dx & dy to caculate the offsets
    total_dx = 0 
    total_dy = 0
    fitted_im1 = im1.copy()
    
    # define the level of the image pyramid (6 or 7)
    # level = 6
    level = 6
    while level >= 0:
        
        # rescale the image with the factor of 2^level
        im1_scaled = sk.transform.resize(fitted_im1, 
                                         (im1.shape[0] // 2**level,
                                          im1.shape[1] // 2**level))
        im2_scaled = sk.transform.resize(im2, 
                                        (im2.shape[0] // 2**level,
                                         im2.shape[1] // 2**level))  
        h = im2_scaled.shape[0]
        w = im2_scaled.shape[1]
        best_x = 0
        best_y = 0
        best_dist = np.inf
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                # shift image
                shifted_im1 = np.roll(np.roll(im1_scaled, i, axis=0), 
                                      j, axis=1)

                # calculate SSD of the central part
                # to avoid the edge affecting the distance
                diff = (shifted_im1-im2_scaled)[h//4:3*h//4, w//4:3*w//4]
                dist = np.sum(diff**2)
        
                if dist < best_dist:
                    best_x = i
                    best_y = j
                    best_dist = dist
        
        dx = (2 ** level) * best_x
        dy = (2 ** level) * best_y
        fitted_im1 = np.roll(np.roll(fitted_im1, dx, axis=0), dy, axis=1)
        
        # update the total dx
        total_dx += dx
        total_dy += dy
        level -= 1
    
    # print offsets (which are total dx and dy)
    print("total x-axis offset of im1 is {}".format(total_dx))
    print("total y-axis offset of im1 is {}".format(total_dy)) 
    return fitted_im1

for img in os.listdir("data"):
    if img.endswith("jpg"):
        generate_output(img, pyramid=False)
    if img.endswith("tif"):
        generate_output(img, pyramid=True)

####################
# Bells and Whistles
####################

# Aligning with Edge Detection
def align_edges_pyramid(im1, im2):
    # save total dx & dy to caculate the offsets
    total_dx = 0 
    total_dy = 0
    fitted_e1 = sk.filters.edges.sobel(im1)
    e2 = sk.filters.edges.sobel(im2)
    
    # define the level of the image pyramid (6 or 7)
    # level = 6
    level = 6
    while level >= 0:
        
        # rescale the image with the factor of 2^level
        e1_scaled = sk.transform.resize(fitted_e1, 
                                        (im1.shape[0] // 2**level,
                                         im1.shape[1] // 2**level))
        e2_scaled = sk.transform.resize(e2, 
                                        (im2.shape[0] // 2**level,
                                         im2.shape[1] // 2**level))  
        h = e2_scaled.shape[0]
        w = e2_scaled.shape[1]
        best_x = 0
        best_y = 0
        best_dist = np.inf
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                # shift image
                shifted_e1 = np.roll(np.roll(e1_scaled, i, axis=0), 
                                     j, axis=1)

                # calculate SSD of the central part
                # to avoid the edge affecting the distance
                diff = (shifted_e1-e2_scaled)[h//4:3*h//4, w//4:3*w//4]
                dist = np.sum(diff**2)
        
                if dist < best_dist:
                    best_x = i
                    best_y = j
                    best_dist = dist
        
        dx = (2 ** level) * best_x
        dy = (2 ** level) * best_y
        fitted_e1 = np.roll(np.roll(fitted_e1, dx, axis=0), dy, axis=1)
        
        # update the total dx
        total_dx += dx
        total_dy += dy
        level -= 1
    
    # print offsets (which are total dx and dy)
    print("total x-axis offset of im1 is {}".format(total_dx))
    print("total y-axis offset of im1 is {}".format(total_dy)) 
    return np.roll(np.roll(im1, total_dx, axis=0), total_dy, axis=1)

# Automatic Contrasting
def contrast(im):
    im_list = []
    for i in range(3):
        curr_im = im[:, :, i]
        upper = curr_im.max()
        lower = curr_im.min()

        # rescale into [0, 1] range 
        # (we use vectorized caculation of numpy)
        im_list.append((curr_im-lower) / (upper-lower))
    
    return np.dstack(im_list)

# Automatic Cropping
def crop_border(im):
    h, w = im.shape[:2]

    edges = [sk.filters.edges.sobel(im[:, :, i]) for i in range(3)]
    
    # set the optimal threshold here (can vary depending on images)
    x_mins = [max(np.argwhere((edge.mean(axis=1)[:h//2] > 0.037))) for edge in edges]
    y_mins = [max(np.argwhere((edge.mean(axis=0)[:w//2] > 0.037))) for edge in edges]

    x_maxs = [h//2 + min(np.argwhere((edge.mean(axis=1)[h//2:] > 0.037))) for edge in edges]
    y_maxs = [w//2 + min(np.argwhere((edge.mean(axis=0)[w//2:] > 0.037))) for edge in edges]

    x_min = min(x_mins)[0]
    y_min = min(y_mins)[0]

    x_max = max(x_maxs)[0]
    y_max = max(y_maxs)[0]
    
    return im[x_min:x_max, y_min:y_max, :]
