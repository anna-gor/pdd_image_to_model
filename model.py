import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab
import cv2
from PIL import Image

def show(img):
    plt.imshow(img)
    plt.show()
    
    
def get_graph(road_markings, imgs_of_signs):
    # посчитаем количество полос
    dividers = ['двойная сплошная', 'сплошная и прерывистая']
    dop_left_line = False
    num_stripes = 0
    if len(road_markings) == 1 and road_markings[0]== 'нет разметки':
        num_stripes = 1
        #print('полос ' + str(num_stripes))
    else:
        for i in range(len(road_markings)):
            line = road_markings[i]
            if line in dividers and i == 0:
                num_stripes += 1
                dop_left_line = True
            if line == 'прерывистая' and i == 0 :
                num_stripes+=1
            if line == 'сплошная' and i == len(road_markings)-1 and not all('сплошная' in line for line in road_markings):
                num_stripes -= 1
            num_stripes += 1 
        print('полос ' + str(num_stripes))

    #Строим сам граф
    R = nx.DiGraph()

    #Расставляем метки знаков
    stripes = []
    sig = len(imgs_of_signs)== 1
    for i in range(num_stripes):
        print(i)
        one_stripe = [0] #сначала пустой участок
        if sig: #делит на 2, т.е. после пустого участка следует участок с действующим знаком
            if dop_left_line and i!=0 or not dop_left_line:
                one_stripe.append(1)
        print(one_stripe)
        stripes.append(one_stripe)
    #print('with signs '+str(stripes))

    R = nx.DiGraph()

    #Определяем распределение вершин по полосам
    i = 0 
    new_obozn = [] #в терминах вершин
    for stripe in stripes:
        listi = []
        for ud in stripe:
            listi.append(i)
            #print(i)
            i+=1
        new_obozn.append(listi)
        #print('next')
    #print('nodes '+ str(new_obozn))

    # Определяем направление движения
    dirs_flg = False
    if sig and not (len(road_markings)%2==1 and road_markings[int(len(road_markings)/2)] in dividers) and road_markings[0] != 'прерывистая':
        #print('simple case with one direction')
        R.add_edges_from(list(map(tuple, [stripe for stripe in new_obozn if len(stripe)>1])), dirs='вперед') #list(map(tuple, [stripe for stripe in new_obozn])) дает таплы полос
        if len(list(map(tuple, [stripe for stripe in new_obozn if len(stripe)==1])))>0:
            R.add_nodes_from([list(map(tuple, [stripe for stripe in new_obozn if len(stripe)==1]))[0][0]])
        dirs_flg = True
    elif sig and road_markings[0] == 'прерывистая' and len(road_markings)>1:
        R.add_edges_from([list(map(tuple, [stripe for stripe in new_obozn]))[0]], dirs='назад')
        R.add_edges_from(list(map(tuple, [stripe for stripe in new_obozn]))[1:], dirs='вперед')
        dirs_flg = True
    else:
        R.add_nodes_from([node[0] for node in new_obozn])

    dirs = list(map(tuple, [stripe for stripe in new_obozn if len(stripe)>1]))
    edge_dirs_labels=dict([((u,v,),d['dirs'])
                     for u,v,d in R.edges(data=True)])

    # Разбираемся с атрибутами вершин, с крайними разметками + знаками
    signed = []
    for i in range(len(stripes)):
        for j in range(len(stripes[i])):
            if stripes[i][j]==1:
                R.nodes[new_obozn[i][j]]["sign"] = 'sign'
                signed.append(new_obozn[i][j])
    #print(signed)

    node_attribute_razm = []
    ind_edge = int(dop_left_line)
    if road_markings[-1]=='сплошная' and not all('сплошная' in line for line in road_markings):
        line = road_markings[-1]
        for i in range(len(stripes[-1])):
            R.nodes[new_obozn[-1][i]]["razm"] = line
            node_attribute_razm.append(new_obozn[-1][i])
    if road_markings[0]=='сплошная' and not all('сплошная' in line for line in road_markings):
        line = road_markings[0]
        for i in range(len(stripes[ind_edge])):
            R.nodes[new_obozn[ind_edge][i]]["razm"] = line
            node_attribute_razm.append(new_obozn[ind_edge][i])

            
    color_map = []
    for node in R:
        if node in signed:
            color_map.append('red')
        else: 
            color_map.append('gray')      

    # Определяем разметку между полосами
    undir_ed = []
    beg = 0
    if dop_left_line:
        for j in range(len(new_obozn[1])):
            R.add_edge(new_obozn[0][0],new_obozn[1][j], mark = road_markings[0])
            undir_ed.append((new_obozn[0][0],new_obozn[1][j]))
        beg = 1
    k=0
    if road_markings[0]!='прерывистая':
        k = 1
    for i in range(beg, len(new_obozn)-1):
        for j in range(len(new_obozn[beg])):
            print('ij', i, j)
            R.add_edge(new_obozn[i][j],new_obozn[i+1][j], mark = road_markings[k])
            undir_ed.append((new_obozn[i][j],new_obozn[i+1][j]))
        k += 1

    between_stripes_marks = []
    for u,v,d in R.edges(data=True):
        if 'mark' in d:
            between_stripes_marks.append(((u,v,),d['mark']))
    edge_between_labels = dict(between_stripes_marks)

    # Окончательно рисуем граф
    pos = nx.spring_layout(R)
    nx.draw_networkx_nodes(R, pos, 
                           node_color=color_map, node_size = 500)
    nx.draw_networkx_labels(R, pos)

    nx.draw_networkx_edge_labels(R,pos,edge_labels=edge_between_labels)
    if dirs_flg:
        nx.draw_networkx_edge_labels(R,pos,edge_labels=edge_dirs_labels)
        nx.draw_networkx_edges(R, pos, edgelist=dirs, arrows=True)
    nx.draw_networkx_edges(R, pos, edgelist=undir_ed, arrows=False)

    pos_higher = {}
    for k, v in pos.items():
        if(v[1]>0):
            pos_higher[k] = (v[0]-0.01, v[1]+0.01)
        else:
            pos_higher[k] = (v[0]-0.01, v[1]-0.01)     
    labels = nx.get_node_attributes(R, 'sign') 
    nx.draw_networkx_labels(R,pos_higher, labels)

    labels1 = nx.get_node_attributes(R, 'razm') 
    pos_lower = {}
    for k, v in pos.items():
        if(v[1]>0):
            pos_lower[k] = (v[0]+0.01, v[1]-0.01)
        else:
            pos_lower[k] = (v[0]+0.01, v[1]+0.01)
    nx.draw_networkx_labels(R,pos_lower, labels1)

    plt.show()
    
    #dict_sign = nx.get_node_attributes(R,'sign')
    #dict_razm = nx.get_node_attributes(R,'razm')
    #dict_mark = nx.get_edge_attributes(R,'mark')
    #dict_dirs = nx.get_edge_attributes(R,'dirs')

    
    return R, imgs_of_signs


def draw_scheme_of_graph(R, imgs_of_signs):  
    def put_sign_on_road(sign_img, img):
        sign_miniature = cv2.resize(sign_img.copy(), (30, 30))
        #show(sign_miniature)
        img[0:sign_miniature.shape[0],0:sign_miniature.shape[1],:] = sign_miniature[0:sign_miniature.shape[0],0:sign_miniature.shape[1],:]
        img = cv2.copyMakeBorder(
                         img, 
                         1, 
                         1, 
                         0, 
                         0, 
                         cv2.BORDER_CONSTANT, 
                         value=(0,0,255)
                      )
        return img

    def sploshnaya_x(img, right = False, left = False, b = 0):
        x = 0
        if left:
            x=b #int(img.shape[1]/6)
        if right:
            x=img.shape[1] - b - 1 #5*int(img.shape[1]/6)
        img = cv2.line(img,  (x, 0), (x, img.shape[0]), (255,255,255), 1)
        return img

    def empty(img, right = False, left = False):
        return img

    def prepyvistaya_x(img, right = False, left = False, b = 0):
        x1=x2=0
        if right:
            x1 = x2 = (int(img.shape[1])-1 - b)
        if left:
            x1 = x2 = b
        step = int(img.shape[0]/6)
        y1 = 0
        for i in range(10):
            y1 = i*step
            y2 = y1+int(step/2)
            img = cv2.line(img,(x1,y1),(x2,y2),(255,255,255),1)
        return img

    def double_sploshnaya_prer(img, right = False, left = False):
        x = 0
        if left:
            x=int(img.shape[1]/6)
        if right:
            x=5*int(img.shape[1]/6)
        img = cv2.line(img,  (x, 0), (x, img.shape[0]), (255,255,255), 1)
        #if left:
        #    img = cv2.line(img,  (x-2, 0), (x-2, img.shape[0]), (255,255,255), 1)
        #if right:
        x = x+2
        #img = cv2.line(img,  (x+2, 0), (x+2, img.shape[0]), (255,255,255), 1)
        #return img
        x1=x2=x
        step = int(img.shape[0]/6)
        y1 = 0
        for i in range(10):
            y1 = i*step
            y2 = y1+int(step/2)
            img = cv2.line(img,(x1,y1),(x2,y2),(255,255,255),1)
        return img

    def double_sploshnaya_x(img, right = False, left = False):
        x = 0
        if left:
            x=int(img.shape[1]/6)
        if right:
            x=5*int(img.shape[1]/6)
        img = cv2.line(img,  (x, 0), (x, img.shape[0]), (255,255,255), 1)
        if left:
            img = cv2.line(img,  (x-2, 0), (x-2, img.shape[0]), (255,255,255), 1)
        if right:
            img = cv2.line(img,  (x+2, 0), (x+2, img.shape[0]), (255,255,255), 1)
        return img

    types_of_lines = {'сплошная':sploshnaya_x, 'прерывистая': prepyvistaya_x, 'двойная сплошная': double_sploshnaya_x, 'сплошная и прерывистая': double_sploshnaya_prer, 'нет разметки':empty} 

    #print('Дорожные участки')
    dict_sign = nx.get_node_attributes(R,'sign')
    dict_razm = nx.get_node_attributes(R,'razm')
    dict_mark = nx.get_edge_attributes(R,'mark')
    dict_dirs = nx.get_edge_attributes(R,'dirs')
    dict_ud = dict()
    for i in R.nodes:
        img=np.zeros((50,40,3),np.uint8)
        dict_ud[i] = img

    if len(imgs_of_signs)>0:
        sign_img = imgs_of_signs[0]
    else:
        sign_img = None

    def get_reverse_dir(d, node):
        for k, v in d.items():
            if v == 'назад' and node in k:
                return True

    for node in dict_ud:
        #print(node)
        if node in dict_razm:# and dict_razm[node] == 'сплошная':
        #print(dict_razm[node])
            if node < len(dict_ud)/2:
                dict_ud[node] = types_of_lines[dict_razm[node]](dict_ud[node], left = True)
            else:
                dict_ud[node] = types_of_lines[dict_razm[node]](dict_ud[node], right = True)
        if node in dict_sign and not get_reverse_dir(dict_dirs,node):
            dict_ud[node] = put_sign_on_road(sign_img, dict_ud[node])
        for m in dict_mark:
            if node in m:
                #print(dict_mark[m])
                #print(m)
                vers = set(m)
                #print(vers)
                vers.remove(node)
                #print(vers)
                another_ver = vers.pop()
                #print(another_ver)                
                if another_ver > node:
                    #dict_ud[node] = prepyvistaya_x(dict_ud[node], right = True)
                    if 'сплошная' in dict_mark[m] and dict_mark[m] == 'двойная сплошная':
                        dict_ud[node] = types_of_lines['сплошная'](dict_ud[node], right = True, b = 1)
                    elif dict_mark[m] == 'сплошная и прерывистая':
                        dict_ud[node] = types_of_lines['сплошная'](dict_ud[node], right = True, b = 1)
                    else:
                        dict_ud[node] = types_of_lines[dict_mark[m]](dict_ud[node], right = True)
                else:
                    if 'сплошная' in dict_mark[m] and dict_mark[m] == 'двойная сплошная':
                        dict_ud[node] = types_of_lines['сплошная'](dict_ud[node], left = True, b = 1)
                    #dict_ud[node] = prepyvistaya_x(dict_ud[node], left = True)
                    elif dict_mark[m] == 'сплошная и прерывистая':
                        dict_ud[node] = types_of_lines['прерывистая'](dict_ud[node], left = True, b = 1)
                    else:
                        dict_ud[node] = types_of_lines[dict_mark[m]](dict_ud[node], left = True)

        #show(dict_ud[node])
    stripes = []
    if len(R.nodes) != len([key for key in dict_dirs])*2: # то есть есть доп линия
        stripes.append(np.concatenate((dict_ud[0], dict_ud[0]), axis=0))
    if len(dict_dirs) != 0:
        for my_dir in dict_dirs:
            stripes.append(np.concatenate((dict_ud[max(my_dir)], dict_ud[min(my_dir)]), axis=0))
    else:
        stripes = [dict_ud[key] for key in dict_ud]

    result = stripes[0]
    for i in range(1, len(stripes)):
        #result = result[0:min(result.shape[0],imgs_of_signs[i].shape[0]),]
        result = np.concatenate((result, stripes[i][0:result.shape[0],0:result.shape[1]]), axis=1) 

    print("Результат")
    show(result)
    
    
