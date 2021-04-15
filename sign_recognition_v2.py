import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show(img):
    plt.imshow(img)
    plt.show()
    
import webcolors

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def sign(imga, color, dop_area = False, mask = None):
    sign_flag=False
    img=cv2.imread(imga)
    target_color = color
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if mask is None:
        if target_color=='red':
            # lower mask (0-10)
            lower_red = np.array([0,50,50])
            upper_red = np.array([10,255,255])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

            # upper mask (170-180)
            lower_red = np.array([170,50,50])
            upper_red = np.array([180,255,255])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
            mask = mask0+mask1

        if target_color=='blue':
            # lower mask (0-10)
            lower_blue = np.array([84,100,100])
            upper_blue = np.array([104,255,255])
            mask2 = cv2.inRange(img_hsv, lower_blue, upper_blue)

            #upper mask (170-180)
            lower_blue1 = np.array([110,50,50])
            upper_blue1 = np.array([130,255,255])
            mask3 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)

            lower_blue = np.array([124,50,50])
            upper_blue = np.array([144,255,255])
            mask4 = cv2.inRange(img_hsv, lower_blue, upper_blue)

            # join my masks
            mask = mask2+mask3+mask4
    #show(mask)
    font = cv2.FONT_HERSHEY_COMPLEX
    img = cv2.imread(imga, cv2.IMREAD_GRAYSCALE)
    threshold = mask
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        img = cv2.imread(imga, cv2.IMREAD_GRAYSCALE)
        cv2.drawContours(img, [approx], 0, (0), 5)
        #x = approx.ravel()[0]
        #y = approx.ravel()[1]
        #show(img)
    max_count = []
    max_len = 0
    for count in contours:
        x, y, w, h = cv2.boundingRect(count)
        if len(count)>max_len and w/h<1.5 and h/w<2:
            max_len = len(count)
            max_count = count
    img = cv2.imread(imga)
    try: 
        cv2.drawContours(img, [max_count], 0, (0), 5)
        #show(img)
        image=cv2.imread(imga)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x, y, w, h = cv2.boundingRect(max_count)
        if dop_area:
            y_new, h_new = y+h, int(h/2)
            x_new, w_new = x, w
            #show(rgb_img[y+h:y+h+int(h/2),x:x+w])
    except:
        return None
    
    col = 0
    if target_color == 'blue':
        for i in range(w):
            for j in range(h):
                if 'blue' in closest_colour(rgb_img[y+j][x+i]) or 'violet' in closest_colour(rgb_img[y+j][x+i]):
                    col+=1
        if col/(w*h)>0.2 and h > 30:
            #show(rgb_img[y:y+h,x:x+w])
            x_coord = int((2*max_count.ravel()[0]+w)/2)
            y_coord = int((2*max_count.ravel()[1]+h)/2)
            sign_flag=True
            #print(x,y)  
            cv2.fillPoly(mask, pts =[max_count], color=(0,0,0))
            #show(mask)
            return mask, rgb_img[y:y+h,x:x+w], (x_coord,y_coord)
        else:
            #print("Sorry, no sign found")
            return None
    if target_color == 'red':
        for i in range(w):
            for j in range(h):
                if 'ros' in closest_colour(rgb_img[y+j][x+i]) or 'pink' in closest_colour(rgb_img[y+j][x+i]) or 'red' in closest_colour(rgb_img[y+j][x+i]) or 'tomato' in closest_colour(rgb_img[y+j][x+i]):
                    col+=1
        #print(col/(w*h))
        if col/(w*h)>0.12 and h > 30:
            #show(rgb_img[y:y+h,x:x+w])
            x_coord = int((2*max_count.ravel()[0]+w)/2)
            y_coord = int((2*max_count.ravel()[1]+h)/2)
            sign_flag=True
            cv2.fillPoly(mask, pts =[max_count], color=(0,0,0))
            #print(x,y)
            return mask, rgb_img[y:y+h,x:x+w], (x_coord,y_coord)
        else:
            #print("Sorry, no sign found")
            return None
    if dop_area and sign_flag:
        #x_new, y_new, w_new, h_new = x, y+h, w, int(h/2)
        col = 0
        for i in range(w_new):
            for j in range(h_new):
                if 'gray' in closest_colour(rgb_img[y_new+j][x_new+i]) or 'white' in closest_colour(rgb_img[y_new+j][x_new+i]) or 'snow' in closest_colour(rgb_img[y_new+j][x_new+i]) or 'black' in closest_colour(rgb_img[y_new+j][x_new+i]):
                    col+=1
        if col/(w_new*h_new)>0.4:
            show(rgb_img[y_new:y_new+h_new,x_new:x_new+w_new])
            
            
def get_imgs_of_signs(test_img):
    list_of_signs = []
    imgs_of_signs = []
    for color in ['red', 'blue']:
        mask = None
        s = sign(test_img, color)
        while True:
            if s is not None:
                list_of_signs.append((s[1], s[2]))
                s = sign(test_img, color, mask = s[0])
            else:
                break
        if len(list_of_signs)== 0:
            continue
        if len(list_of_signs)>1:
            list_of_signs = sorted(list_of_signs, key = lambda k : (k[1][0]))
            a = np.array([sign[1][1] for sign in list_of_signs])
            min_diff = np.diff(np.unique(a)).min()
            if min_diff < 4:
                imgs_of_signs = [sign[0] for sign in list_of_signs]
                #print(len(imgs_of_signs))
                result = imgs_of_signs[0]
                #show(result)
                #print('begin')
                for i in range(1, len(imgs_of_signs)):
                    result = result[0:min(result.shape[0],imgs_of_signs[i].shape[0]),]
                    #show(result)
                    result = np.concatenate((result, imgs_of_signs[i][0:result.shape[0],0:result.shape[1]]), axis=1) 
                imgs_of_signs = [result]
                #show(list_of_signs[0])
        else:
            imgs_of_signs=[sign[0] for sign in list_of_signs]
    return imgs_of_signs