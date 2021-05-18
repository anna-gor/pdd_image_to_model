import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show(img):
    plt.imshow(img)
    plt.show()
    
def get_middle_point_for_lines(test_mask, point):
    x, y = point
    white_range = []
    average = 0
    if test_mask[y][x] == 0:
        for j in reversed(range(x)):
            if test_mask[y][j] == 0:
                continue
            else:
                #print(y,j)
                white_range.append(j)
                average = average + j
                break
        for j in range(x, test_mask.shape[1]):
            if test_mask[y][j] == 0:
                continue
            else:
                #print(y1,j)
                white_range.append(j)
                average = average + j
                break
    return int(average/2), y

def find_number_of_lines(test_mask_clear, ratio): # среди точек на перечесении серединной прямой и белой области находим точки в серединах белых областей
    dic={}
    i = int(ratio*test_mask_clear.shape[0]/100)
    for j in range(len(test_mask_clear[i])):
        if test_mask_clear[i][j] not in dic:
            dic[test_mask_clear[i][j]] = [j]
        dic[test_mask_clear[i][j]].append(j)
    if 0 not in dic:
        return
    whites = dic[0]
    points = []
    b = 0
    for p in range(len(dic[0])-1):
        if p == len(dic[0])-2:
            points.append((whites[int((b+p+1)/2)], i))
        elif whites[p+1]-whites[p]<=1:
            continue
        else:
            points.append((whites[int((b+p)/2)], i))
            b = p+1
    return points

def contrast_of_img(image1, brightness = -150, contrast = 150):
    img = np.int16(image1)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return cv2.medianBlur(img,5)

def get_full_mask(test_img):
    image = cv2.imread(test_img) 
    beg_contrast_img = contrast_of_img(image)
    else_contrast_img = contrast_of_img(beg_contrast_img, brightness = 150, contrast = 150) 
    contrast_img = contrast_of_img(else_contrast_img, brightness = -30, contrast = 150) 
    hsv = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2HSV )
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([15, 15, 15], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    width, height = mask.shape
    full_mask = mask[int(width/2):width, ]
    return full_mask

def crop_thresh(contrast_img):
    gray = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    width, height = contrast_img.shape[:-1]
    img = thresh[int(width/2):width, ]
    return img

def get_clean_mask(test_img):
    image = cv2.imread(test_img) 
    beg_contrast_img = contrast_of_img(image)
    else_contrast_img = contrast_of_img(beg_contrast_img, brightness = 150, contrast = 150) 
    contrast_img = contrast_of_img(else_contrast_img, brightness = -30, contrast = 150) 
    clean_mask = crop_thresh(contrast_img) #то же самое что и cropped masks
    return clean_mask

def get_correct_lines(test_mask,points):
    #определили координаты маски (белых участков)
    wh=[]
    for i in range(len(test_mask)):
        for j in range(len(test_mask[0])):
            if test_mask[i][j]==0:
                wh.append((j,i))
                
    def find_eps(point): # тут можно нарисовать схемку, почему это так, что имеется в виду
        x, y = point
        up_line = 0
        down_line = 0
        for j in list(range(point[1], test_mask.shape[0])):#идем по вертикали вниз
            if test_mask[j][x]!=255:
                up_line += 1
            else:
                break
        for j in list(range(point[1], 0, -1)): #идем по вертикали вверх
            if test_mask[j][x]!=255:
                down_line += 1
            else:
                break
        if min(up_line,down_line)>3:
            return min(up_line,down_line)
        return 3
    
    def coords_of_line(p1,p2):
        coords = []
        if p2[1]-p1[1] != 0:
            k=(p2[0]-p1[0])/(p2[1]-p1[1])
            b = p1[0] - k*p1[1]
        #print(k,b)
            for x in range(0, test_mask.shape[0]):
                y=int(k*x+b)
                coords.append((y,x))
            return coords
        
    def find_line_for_razm(line, point):
        wh=[]
        for i in range(len(test_mask)):
            for j in range(len(test_mask[0])):
                if test_mask[i][j]==0:
                    wh.append((j,i))
        mse = get_MSE(line)
        len_line = len(line)
        intersect = len(list(set(wh) & set(line)))
        for ind in list(range(point[1]+3, len_line, 10)):
            try:
                prob_line = coords_of_line(get_middle_point(line[point[1]]),get_middle_point(line[ind]))
                mse_prob = get_MSE(prob_line)
                prob_intersect = len(list(set(wh) & set(prob_line)))
                if mse_prob<mse and prob_intersect>=intersect:
                    mse=mse_prob
                    line=prob_line
                    intersect = prob_intersect
            except:
                pass
        for ind in list(range(point[1]-3, 0 , -10)):
            try:
                prob_line = coords_of_line(get_middle_point(line[point[1]]),get_middle_point(line[ind]))
                mse_prob = get_MSE(prob_line)
                prob_intersect = len(list(set(wh) & set(prob_line)))
                if mse_prob<mse and prob_intersect>=intersect:
                    mse=mse_prob
                    line=prob_line
                    intersect = prob_intersect
            except:
                pass
        return line
    
    def point_MSE(point):
        x, y = point
        white_range = []
        average = 0
        left = 0
        right = 0
        if test_mask[y][x] == 0:
            for j in reversed(range(x)):
                if test_mask[y][j] == 0:
                    left +=1
                    continue
                else:
                    #print(y,j)
                    white_range.append(j)
                    average = average + j
                    break
            for j in range(x, test_mask.shape[1]):
                if test_mask[y][j] == 0:
                    right += 1
                    continue
                else:
                    #print(y,j)
                    white_range.append(j)
                    average = average + j
                    break
        return abs(right - left)

    def get_MSE(line1):
        mse = 0
        metric_idxs = [i for i in range(0,len(line1),10)]
        metric_points = [line1[i] for i in metric_idxs]
        for point in metric_points:
            mse+=point_MSE(point)
        return mse


    up_and_down_points = []
    for point in points:
        eps = find_eps(point)
        x,y = point
        up_and_down = [(i,j) for i in range(point[0]-20,point[0]+20) for j in [y-eps, y+eps]]
        my_points = [p for p in up_and_down if p in wh]
        up_and_down_points.append(my_points)
    
    
    direction_lines = []
    for i in range(len(points)):
        metric = 0
        line = []
        for c in up_and_down_points[i]:
            my_line = coords_of_line(points[i],c)
            my_metric = len(list(set(wh) & set(my_line)))
            if my_metric>metric:
                line = my_line 
                metric = my_metric
        direction_lines.append(line)
        
    correct_lines = [find_line_for_razm(direction_lines[i], points[i]) for i in range(len(points))]
        
    return correct_lines

def get_road_markings(test_img):
    img = cv2.imread(test_img, 1)
    height, width = img.shape[:2] 
    crop_img = img[int(height / 2):height, 0:width]
    show(crop_img)
    com_masks = []
    com_bad_masks = []
    full_mask = get_full_mask(test_img)
    clean_mask = get_clean_mask(test_img)
    thresh = np.invert(clean_mask)
    print('NEW')
    connectivity = 4  
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    for i in range(0, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        output = crop_img.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        componentMask = (labels == i).astype("uint8") * 255
        mask_points = []
        for i in range(len(componentMask)):
            for j in range(len(componentMask[0])):
                if componentMask[i][j]==255:
                    mask_points.append((j,i))
        dict_mask = {}
        for point in mask_points:
            if point[1] not in dict_mask:
                dict_mask[point[1]] = [point]
            else:
                dict_mask[point[1]].append(point)

        leng = len(dict_mask)
        max_wid = 0
        avg_wid = 0
        for key in dict_mask:
            my_wid = abs(dict_mask[key][0][0]-dict_mask[key][-1][0])
            if (my_wid>max_wid):
                max_wid=my_wid
            avg_wid += abs(dict_mask[key][0][0]-dict_mask[key][-1][0])
        wid = max_wid
        av_wid = avg_wid/len(dict_mask)
        #анализируем угол наклона
        test_mask_inv = np.invert(componentMask)
        test_mask = test_mask_inv

        points_to_analyze = []
        for ratio in range(1,99):
            if find_number_of_lines(test_mask, ratio):
                points_to_analyze.append(find_number_of_lines(test_mask, ratio))
        try:
            p1=points_to_analyze[0][0]
            p2=points_to_analyze[-1][0]
            try:
                k=(p2[0]-p1[0])/(p2[1]-p1[1])
            except:
                k=0
        except:
                k=0

        if (leng>max_wid and max_wid< 100) and not (k>0 and points_to_analyze[int(len(points_to_analyze)/2)][0][0]<test_mask.shape[1]/2) and len(points_to_analyze)>3 and not (k<0 and points_to_analyze[int(len(points_to_analyze)/3)][0][0]>2*test_mask.shape[1]/3) and points_to_analyze[-1][0][1]>18:
            print(len(points_to_analyze))
            show(output)
            show(componentMask)
            com_masks.append(componentMask)
        else:
            com_bad_masks.append(componentMask)

    # очистим маску
    noiseless_mask = full_mask.copy()
    for bad_mask in com_bad_masks[1:]:
        noiseless_mask = noiseless_mask + bad_mask
    show(noiseless_mask)

    #достанем ключевые точки
    points = []
    for i in range(len(com_masks)):
        test_mask = np.invert(com_masks[i])
        in_mask = test_mask
        points_to_analyze = []
        for ratio in range(1,99):
            if find_number_of_lines(in_mask, ratio):
                points_to_analyze.append(find_number_of_lines(in_mask, ratio))
        two_lines = [i for i in points_to_analyze if len(i)==2]
        if (len(two_lines) >0):
            for point in two_lines[int(len(two_lines)/2)]:
                points.append(get_middle_point_for_lines(in_mask, point))
        else:
            points.append(get_middle_point_for_lines(in_mask, points_to_analyze[int(len(points_to_analyze)/2)][0]))

    #строим линии
    test_mask =  noiseless_mask
    
    image = crop_img.copy()
    print(points)
    show(test_mask)
    my_correct_lines = []
    for point in points:
        try:
            my_correct_lines.append(get_correct_lines(test_mask,[point]))
        except:
            pass
    for line in my_correct_lines:
        for line1 in my_correct_lines:
            if len(list(set(line[0]) & set(line1[0])))>10 and len(list(set(line[0]) & set(line1[0])))<len(line[0]):
                index = my_correct_lines.index(line1)
                points.remove(points[index])
                my_correct_lines.remove(line1)
    for i in range(len(my_correct_lines)):
        image = cv2.line(image, my_correct_lines[i][0][0],my_correct_lines[i][0][-1],(0, 0, 0), 5)
    print(len(my_correct_lines))
    print(len(points))
    show(image)

    # Распознаем разметку
    
    def coords_of_line(p1,p2):
        coords = []
        if p2[1]-p1[1] != 0:
            k=(p2[0]-p1[0])/(p2[1]-p1[1])
            b = p1[0] - k*p1[1]
        #print(k,b)
            for x in range(0, test_mask.shape[0]):
                y=int(k*x+b)
                coords.append((y,x))
            return coords
        
    def get_middle_point(point):
        y, x = point
        white_range = []
        average = 0
        if test_mask[y][x] != 255:
            for j in reversed(range(x)):
                if test_mask[y][j] != 255:
                    continue
                else:
                    #print(y,j)
                    white_range.append(j)
                    average = average + j
                    break
            for j in range(x, test_mask.shape[1]):
                if test_mask[y][j] != 255:
                    continue
                else:
                    #print(y1,j)
                    white_range.append(j)
                    average = average + j
                    break
        return int(average/2), y

    def get_per_point(p, matrix):
        pxu = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
        pyu = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
        pu_after = (int(pxu), int(pyu))
        return pu_after

    def point_MSE(point):
        x, y = point
        white_range = []
        average = 0
        left = 0
        right = 0
        if test_mask[y][x] == 0:
            for j in reversed(range(x)):
                if test_mask[y][j] == 0:
                    left +=1
                    continue
                else:
                    #print(y,j)
                    white_range.append(j)
                    average = average + j
                    break
            for j in range(x, test_mask.shape[1]):
                if test_mask[y][j] == 0:
                    right += 1
                    continue
                else:
                    #print(y,j)
                    white_range.append(j)
                    average = average + j
                    break
        return abs(right - left)

    def get_MSE(line1):
        mse = 0
        metric_idxs = [i for i in range(0,len(line1),10)]
        metric_points = [line1[i] for i in metric_idxs]
        for point in metric_points:
            mse+=point_MSE(point)
        return mse
    
    def get_middle_point(point):
        x, y = point
        white_range = []
        average = 0
        if test_mask[y][x] == 0:
            for j in reversed(range(x)):
                if test_mask[y][j] == 0:
                    continue
                else:
                    #print(y,j)
                    white_range.append(j)
                    average = average + j
                    break
            for j in range(x, test_mask.shape[1]):
                if test_mask[y][j] == 0:
                    continue
                else:
                    #print(y,j)
                    white_range.append(j)
                    average = average + j
                    break
        return int(average/2), y
    
    def get_per_line(line, key_point, matrix):
        p1 = get_per_point(line[0], matrix)
        p2 = get_per_point(key_point, matrix)
        line = coords_of_line(p1,p2)
        return line
    
    def find_line_for_razm(line, point):
        mse = get_MSE(line)
        len_line = len(line)
        intersect = len(list(set(wh) & set(line)))
        for ind in list(range(point[1]+3, len_line, 10)):
            try:
                prob_line = coords_of_line(get_middle_point(line[point[1]]),get_middle_point(line[ind]))
                mse_prob = get_MSE(prob_line)
                prob_intersect = len(list(set(wh) & set(prob_line)))
                if mse_prob<mse and prob_intersect>=intersect:
                    mse=mse_prob
                    line=prob_line
                    intersect = prob_intersect
            except:
                pass
        for ind in list(range(point[1]-3, 0 , -10)):
            try:
                prob_line = coords_of_line(get_middle_point(line[point[1]]),get_middle_point(line[ind]))
                mse_prob = get_MSE(prob_line)
                prob_intersect = len(list(set(wh) & set(prob_line)))
                if mse_prob<mse and prob_intersect>=intersect:
                    mse=mse_prob
                    line=prob_line
                    intersect = prob_intersect
            except:
                pass
        return line
    
    def find_len_line(line, point, colors):
        max_len = 0
        #colors=[test_mask[y][x] for x, y in line]
        len_line = 0
        for ind in list(range(point[1], len(line))):
            if colors[ind]!=255:
                len_line += 1
            else:
                break
        for ind in list(range(point[1], 0, -1)):
            if colors[ind]!=255:
                len_line += 1
            else:
                break
        return len_line


    all_stripes = []
    correct_lines = [line[-1] for line in my_correct_lines]
    correct_lines = sorted(correct_lines, key = lambda k : (k[-1][0]))
    points = sorted(points, key = lambda k : (k[0]))
    test_mask_before = noiseless_mask.copy()
    test_img = crop_img.copy()
    if len(correct_lines) == 0:
        show(test_img)
        all_stripes.append((('нет разметки'), 0))
        print('нет разметки')
        return ['нет разметки']
    
    b=10
    lu = correct_lines[0][int(len(correct_lines[0])/4)] #black 1
    ld = correct_lines[0][-1] #red 3
    ru = correct_lines[-1][int(len(correct_lines[-1])/4)] #green 2
    rd = correct_lines[-1][-1] #blue 4
    src = np.float32([(lu[0]-b, 0),
                  (ru[0]+b, ru[1]),
                  (ld[0]-b, ld[1]),
                  (rd[0]+b, rd[1])])

    dst = np.float32([(0, 0),
                      (500, 0),
                      (0, 600),
                      (500, 600)])
    matrix = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(test_img, matrix, (500, 600))
    show(result)
    test_mask = noiseless_mask.copy()
    test_mask = cv2.warpPerspective(test_mask, matrix, (500, 600))
    #show(test_mask)
    wh=[]
    for i in range(len(test_mask)):
        for j in range(len(test_mask[0])):
            if test_mask[i][j]==0:
                wh.append((j,i))
    for i in range(len(correct_lines)):
        line = get_per_line(correct_lines[i], points[i], matrix)
        point = get_per_point(points[i], matrix)
        find_line = find_line_for_razm(line, point)
        # Определим длину
        colors=[test_mask[y][x] for x, y in line]
        num_stripes = len(find_line)/find_len_line(find_line, point, colors)
        if num_stripes<2:
            all_stripes.append((('сплошная'), find_line[int(len(find_line)/2)][0]))
            #print( ('сплошная'), find_line[int(len(find_line)/2)][0])
        else:
            all_stripes.append((('прерывистая'), find_line[int(len(find_line)/2)][0]))
            #print( ('прерывистая'), find_line[int(len(find_line)/2)][0])
   
    print(all_stripes)
    final_answer = []
    used_lines = set()
    a = np.array([tup[1] for tup in all_stripes])
    max_dist = result.shape[1]
    min_diff = np.diff(np.unique(a)).min()
    #print(min_diff)
    #print(result.shape)
    for i in range(1, len(all_stripes)):
        if len(all_stripes)==1 and all_stripes[0][0]=='нет разметки':
            final_answer.append('нет разметки')
        if abs(all_stripes[i-1][1] - all_stripes[i][1])<max_dist/2 and abs(all_stripes[i-1][1] - all_stripes[i][1])<1.1*min_diff:
            if all_stripes[i-1] in used_lines:
                final_answer.remove(all_stripes[i-1][0])
            else:
                used_lines.add(all_stripes[i-1])
            used_lines.add(all_stripes[i])
            if all_stripes[i-1][0] == all_stripes[i][0]:
                final_answer.append('двойная '+ all_stripes[i-1][0])
            else:
                final_answer.append(all_stripes[i-1][0]+ ' и ' + all_stripes[i][0])
        else:
            if all_stripes[i-1] not in used_lines:
                final_answer.append(all_stripes[i-1][0])
                used_lines.add(all_stripes[i-1])
            if all_stripes[i] not in used_lines:
                final_answer.append(all_stripes[i][0])
                used_lines.add(all_stripes[i])
    print(final_answer)
    return final_answer

