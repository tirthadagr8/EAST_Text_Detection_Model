from shapely.geometry import Polygon
import numpy as np
import cv2
from PIL import Image
import math
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm
import json

from shapely.geometry import Polygon
import numpy as np
import cv2
from PIL import Image
import math
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data


def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:	
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x) 
        vertices[y1_index] += ratio * (-length_y) 
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x 
        vertices[y2_index] += ratio * length_y
    return vertices	


def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
    r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
    r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
    r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
        offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:	
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:,:1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err	


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list: 
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k : area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
        True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, \
          start_w + length, start_h + length, start_w, start_h + length]).reshape((4,2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4,2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99: 
            return True
    return False


def crop_img(img, vertices, labels, length):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
        print('Done 1')
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
        print('Done 2')
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, new_vertices


def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices

from PIL import Image
import numpy as np

from PIL import Image
import numpy as np

def resize_img_and_vertices(img, vertices, target_size):
    '''Resize image to a square shape and adjust vertices correspondingly.
    Input:
        img         : PIL Image object
        vertices    : Vertices of text regions <numpy.ndarray, (n,8)>
        target_size : The target square size (e.g., 2048 for 2048x2048)
    Output:
        resized_img : Resized square image (PIL Image)
        new_vertices: Adjusted vertices for the resized image <numpy.ndarray, (n,8)>
    '''
    # Get the original dimensions of the image
    orig_w, orig_h = img.width, img.height

    # Resize the image to a square of target_size x target_size
    resized_img = img.resize((target_size, target_size), Image.BILINEAR)

    # Calculate the scaling ratios for both width and height
    ratio_w = target_size / orig_w
    ratio_h = target_size / orig_h

    # Adjust the vertices by scaling the x and y coordinates
    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w  # x-coordinates
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h  # y-coordinates

    return resized_img, new_vertices



def get_score_geo(img, vertices, labels, scale, length):
    '''generate score gt and geometry gt
    Input:
        img     : PIL Image
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        scale   : feature map / image
        length  : image length
    Output:
        score gt, geo gt, ignored
    '''
    score_map   = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    geo_map     = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
    ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)

    index = np.arange(0, length, int(1/scale))
    index_x, index_y = np.meshgrid(index, index)
    ignored_polys = []
    polys = []

    for i, vertice in enumerate(vertices):
        if labels[i] == 0:
            ignored_polys.append(np.around(scale * vertice.reshape((4,2))).astype(np.int32))
            continue

        poly = np.around(scale * shrink_poly(vertice).reshape((4,2))).astype(np.int32) # scaled & shrinked
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)

        theta = find_min_rect_angle(vertice)
        rotate_mat = get_rotate_mat(theta)

        rotated_vertices = rotate_vertices(vertice, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)

        d1 = rotated_y - y_min
        d1[d1<0] = 0
        d2 = y_max - rotated_y
        d2[d2<0] = 0
        d3 = rotated_x - x_min
        d3[d3<0] = 0
        d4 = x_max - rotated_x
        d4[d4<0] = 0
        geo_map[:,:,0] += d1[index_y, index_x] * temp_mask
        geo_map[:,:,1] += d2[index_y, index_x] * temp_mask
        geo_map[:,:,2] += d3[index_y, index_x] * temp_mask
        geo_map[:,:,3] += d4[index_y, index_x] * temp_mask
        geo_map[:,:,4] += theta * temp_mask

    cv2.fillPoly(ignored_map, ignored_polys, 1)
    cv2.fillPoly(score_map, polys, 1)
    return torch.Tensor(score_map).permute(2,0,1), torch.Tensor(geo_map).permute(2,0,1), torch.Tensor(ignored_map).permute(2,0,1)


def extract_vertices(lines):
    '''extract vertices info from txt lines
    Input:
        lines   : list of string info
    Output:
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
    '''
    labels = []
    vertices = []
    for line in lines:
        vertices.append(list(map(int,line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
        label = 0 if '###' in line else 1
        labels.append(label)
    return np.array(vertices), np.array(labels)

def get_metadata():
	json_filepaths=[]
	for path,_,names in os.walk('C:/Users/tirth/Desktop/manga_text_ocr-main'):
		for name in names: 
			if '.json' in name:
				json_filepaths.append(os.path.join(path,name))
				
	max_tokens=128
	unique_characters=set()
	metadata=[]
	for path in tqdm(json_filepaths):
		with open(path,'r',encoding='utf8') as f:
			data=json.load(f)
		texts=[]
		txt=[]
		coords=[]
		for annot in data['shapes']:
			unique_characters.update(annot['label'])
			txt.append(annot['label'])
			# x1=annot['points'][0][0] y1=annot['points'][0][1] x2=annot['points'][1][0] y2=annot['points'][1][1]
			coords.append([annot['points'][0][0],annot['points'][0][1],annot['points'][1][0],annot['points'][0][1],annot['points'][1][0],annot['points'][1][1],annot['points'][0][0],annot['points'][1][1]])
		texts=(' ').join(txt)
		if os.path.exists(path.replace('json','png')):
			img_path=path.replace('json','png')
		elif os.path.exists(path.replace('json','jpg')):
			img_path=path.replace('json','jpg')
		metadata.append((img_path,texts,coords))
	unique_characters=['#',' ']+sorted(unique_characters)
	vocab=('').join(sorted(unique_characters))
	vocab=vocab+' !"#$%\'()*+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]`abcdefghiklmnopqrstuvwxyz{|}°ΛΣαβγδεηθλμνξπσφω‥…※①②③④⑤⑥⑦□▲△○●、。〃々〆〇「」『』【】〓〔〕〳〴〵〻あぃいうえおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわゐゑをんゔゝゞゟァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶ・ーヽヾヿ㊀㊁㊂㊃㊄㊅㊆㊇㊈㊉㊞一丁七万丈三上下不与丐丑且世丗丘丙丞両並个中丱串丸丹主丼乂乃久之乍乎乏乖乗乘乙九乞也乱乳乾亀亂了予争亊事二于云互五井亘亙些亜亞亟亡亢交亥亦亨享京亭亮亰亳人什仁仂仄仆仇今介仍从仏仔仕他仗付仙仟代令以仭仮仰仲件价任企伉伊伍伎伏伐休会伜伝伯伴伸伺似伽佃但佇位低住佐佑体何佗余佚佛作佝佞佩佳併佶佻使侃來侈例侍侑侖侘供依価侫侭侮侯侵侶便係促俄俊俑俗俘俚保俟信俣俤俥修俯俳俵俶俸俺俾倆倉個倍倏們倒倔倖候倚借倡倣値倦倨倩倪倫倭倶倹偃假偈偉偏偕偖做停健偲側偵偶偸偽傀傅傍傑傘備傚催傭傲傳債傷傾僂僅僉僊働像僕僖僚僞僣僥僧僭僮僵價僻儀儂億儉儒儕儘償儡優儲儺儼兀允元兄充兆兇先光克兌免兎児兒兔党兜入全兩八公六兮共兵其具典兼冀内円冉冊册再冏冐冑冒冓冕冗写冠冥冦冨冩冪冬冰冱冲决况冶冷冽凄准凉凋凌凍凖凛凜凝几凡処凪凰凱凶凸凹出函凾刀刃刄分切刈刊刋刎刑列初判別刧利刪刮到制刷券刹刺刻剃則削剋剌前剏剔剖剛剞剣剥剩剪副剰割剳剴創剽剿劃劇劈劉劍劑劒劔力功加劣助努劫励労劵効劾勁勃勅勇勉勒動勗勘務勝勞募勢勤勦勧勲勳勵勸勺勾勿匁匂包匆匈匍匏匐匕化北匙匝匠匡匣匪匳匹区医匿區十千卅升午卉半卑卒卓協南単博卜占卦卯印危即却卵卷卸卻卿厄厘厚原厠厥厦厨厩厭厶去参參又叉及友双反収叔取受叙叛叟叡叢口古句叨叩只叫召叭叮可台叱史右叶号司叺吁吃各合吉吊吋同名后吏吐向君吝吟吠否含听吮吶吸吹吻吼吽吾呀呂呆呈呉告呎呑呟周呪呱味呵呶呷呻呼命咀咄咆咋和咎咏咒咤咨咫咬咯咲咳咸咽哀品哂哇哈哉員哢哥哨哩哭哮哲哺唄唆唇唐售唯唱唸唹唾啄商問啓啖啗啜啣啻啼喀喃善喇喉喊喋喘喙喚喜喝喞喧喩喪喫喬單喰営嗄嗅嗇嗔嗚嗜嗟嗣嗷嗹嗽嗾嘆嘉嘔嘖嘗嘘嘛嘩嘯嘱嘲嘴嘶嘸噂噌噎噐噛噤器噪噫噬噴噸噺嚀嚆嚇嚔嚢嚥嚮嚴嚼囀囁囂囃囑囓囘囚四回因団囮困囲図固国圀圃圈國圍圏園圓圖團圜土圧在圭地址坂均坊坎坏坐坑坡坤坦坩坪垂型垓垢垣垤垰埀埃埋城埒埓埔埜域埠埴執培基埼堀堂堅堆堊堕堝堡堤堪堯堰報場堵堺塀塁塊塋塑塔塗塘塙塚塞塢塩填塲塵塹塾境墓増墜墟墨墮墳墺墻墾壁壅壇壊壌壑壓壕壘壙壜壞壟壤士壬壮壯声壱売壷壹壺壼壽変夏夕外夘夙多夛夜夢夥大天太夫夭央失夲夷夸夾奄奇奈奉奎奏契奔奕套奚奠奢奥奧奨奩奪奬奮女奴奸好妁如妃妄妊妍妓妖妙妝妥妨妬妹妻妾姆姉始姐姑姓委姙姚姜姥姦姨姪姫姻姿威娃娉娑娘娟娠娥娩娯娶娼婆婉婚婢婦婪婬婿媒媚媛媼嫁嫉嫌嫖嫗嫡嫣嫩嫻嬉嬌嬖嬢嬪嬬嬰嬶孃子孑孔孕字存孚孛孜孝孟季孤孥学孩孫孰孱孳孵學孺宀它宅宇守安宋完宍宏宕宗官宙定宛宜宝実客宣室宥宦宮宰害宴宵家宸容宿寂寃寄寅密寇富寐寒寓寔寛寝寞察寡寢寤寥實寧寨審寫寮寳寵寶寸寺対寿封専射尅将將專尉尊尋對導小少尓尖尚尠尢尤尨就尸尹尺尻尼尽尾尿局居屆屈届屋屍屎屏屐屑屓展属屠屡層履屬屯山屹岐岑岡岨岩岫岬岱岳岷岸峙峠峨峩峭峯峰島峻峽崇崎崑崔崕崖崗崙崛崢崩嵌嵐嵜嵩嵯嶂嶄嶇嶋嶌嶮嶺嶼嶽巉巌巍巒巓巖川州巡巣工左巧巨巫差己已巳巴巵巷巻巽巾市布帆希帑帖帙帚帛帝帥師席帯帰帳帶帷常帽幄幅幇幌幔幕幗幟幡幢幣幤干平年幵并幸幹幺幻幼幽幾广庁広庄庇床序底庖店庚府庠度座庫庭庵庶康庸廂廃廈廉廊廐廓廚廟廠廡廢廣廨廩廬廰廳延廷廸建廻廼廾廿弁弃弄弉弊弋弌式弑弓弔引弖弗弘弛弟弥弦弧弭弱張強弼弾彈彊彌彎当彖彗彙彝彡形彦彩彪彫彬彭彰影彳彷役彼彿往征徂徃径待徇徊律後徐徑徒従得徘徙從徠御徨復循徭微徳徴徹徼徽心必忌忍志忘忙応忝忠忤快忰念忸忻忽忿怎怏怒怕怖怛怜思怠怡急性怨怩怪怫怯怱怺恁恃恆恊恋恍恐恒恕恙恚恟恠恢恣恤恥恨恩恪恬恭息恰恵悄悉悋悌悍悔悖悚悛悟悠患悦悧悩悪悲悴悵悶悸悼悽情惑惓惘惚惜惟惠惡惣惧惨惰惱想惴惶惹惻愁愆愈愉愍意愕愚愛感愧愬愴愼愿慂慄慇慈態慌慍慎慓慕慘慙慚慢慣慥慧慨慫慮慰慴慶慷慾憂憇憊憎憐憑憔憖憙憚憤憧憩憫憬憲憶憺憾懃懇懈應懋懌懍懐懣懦懲懶懷懸懺懼懽懾懿戀戈戊戌戍戎成我戒戔或戚戟戦截戮戯戰戲戳戴戸戻房所扁扇扈扉手才扎打払托扞扣扨扮扱扶批扼承技抂抃抄把抑抓抔投抗折抛抜択披抬抱抵抹押抽拂担拆拇拈拉拊拌拍拏拐拒拓拔拗拘拙招拜拝拠括拭拮拱拳拵拶拷拾拿持挂指按挌挑挙挟挨挫振挺挽挾挿捉捌捍捏捐捕捗捜捧捨捩据捲捶捷捺捻掀掃授掉掌掏排掖掘掛掟掠採探掣接控推掩措掬掲掻掾揀揃揆揉描提插揖揚換握揣揮援揺搆損搏搖搗搜搦搬搭搶携搾摂摘摧摩摯摶摸摺撃撈撒撓撕撚撞撤撥撫播撮撰撲撹撻撼擁擂擅擇操擒擔據擠擡擢擣擦擧擬擯擱擲擴擺擾攀攅攘攝攣攪攫攬支攵收攷改攻放政故效敍敏救敕敖敗教敝敞敢散敦敬数敲整敵敷數斂斃文斉斌斐斑斗料斛斜斟斡斤斥斧斫斬断斯新斷方於施旁旃旅旋旌族旗旛无旡既日旦旧旨早旬旭旱旺昂昆昇昊昌明昏易昔昜星映春昧昨昭是昴昵昼晁時晃晉晋晏晒晝晟晤晦晨晩普景晰晴晶智暁暄暇暈暉暑暖暗暢暦暫暮暴暸暹暼暾曁曄曇曉曖曙曜曝曠曩曰曲曳更曷書曹曼曽曾替最會月有朋服朔朕朗望朝期朦朧木未末本札朮朱朴朶机朽朿杆杉李杏材村杓杖杜杞束条杢杣杤来杭杯東杲杳杵杷杼松板枇枉枋析枕林枚果枝枠枡枩枯枳枴架枷枸枹柁柄柎柏某柑染柔柘柚柝柞柢柩柬柯柱柳柴柵査柾柿栂栃栄栓栖栗校栢栩株栲栴核根格栽桀桁桂桃框案桎桐桑桓桝桴桶桷桿梁梃梅梏梓梗條梟梠梢梧梨梭梯械梱梳梵梶梺棄棉棊棋棍棒棕棗棘棚棟棠棣棧森棲棹棺椀椄椅椋植椎椏椒検椰椹椽椿楊楓楔楕楙楚楞楠楡楢楪楫業楮楯楳極楷楹楼楽概榊榎榔榛榜榮榴榻槁槃槇構槌槍槐槓様槧槭槻槽樂樅樋樒樓樗標樞樟模樣権横樫樵樸樹樺樽橄橇橈橋橘橙機橢橦橿檀檄檎檐檜檢檣檬檮檳檸檻櫂櫃櫓櫚櫛櫞櫟櫻欄欅權欒欖欝欠次欣欧欲欸欹欺欽款歃歇歉歌歎歐歓歙歛歟歡止正此武歩歪歯歳歴歸死歿殀殃殄殆殉殊残殖殘殞殪殫殯殲殳段殷殺殻殼殿毀毅毆毋母毎毒比毘毛毫毬毯氈氏民氓気氛氣水氷永氾汀汁求汎汐汕汗汚汝汞江池汪汰汲決汽沁沂沃沈沌沐沒沓沖沙沛没沢沫沮沱河沸油治沼沽沾沿況泄泉泊泌泓法泗泛泡波泣泥注泪泰泳洋洌洒洗洛洞津洩洪洫洲洳洵活洽派流浄浅浙浚浣浦浩浪浬浮浴海浸浹涅消涌涎涓涕涙涯液涵涸涼淀淆淋淑淘淡淤淨淪淫淬淮深淳淵混淹淺添清渇済渉渋渓渙減渝渟渠渡渣渤渥渦温渫測渭港游渺渾湃湊湍湖湘湛湟湧湫湮湯湾湿満溂源準溘溜溝溟溢溥溪溯溶溷溺滂滄滅滉滋滌滑滓滔滝滞滬滯滲滴滷滸滾滿漁漂漆漉漏漑漓演漕漠漢漣漫漬漱漲漸漿潔潘潛潜潟潤潦潭潮潰潴潸潼澀澁澂澄澆澎澗澡澣澤澪澱澳澹激濁濂濃濆濔濕濘濛濟濠濡濤濫濯濱濳濶濾瀁瀉瀋瀏瀑瀕瀘瀝瀞瀟瀦瀧瀬瀰瀾灌灑灘灣火灯灰灸灼災炉炊炎炒炙炬炭炮炳点為烈烏烙烝烟烱烹烽焉焔焙焚無焦然焼煉煌煎煕煖煙煢煤煥煦照煩煮煽熄熈熊熏熔熙熟熬熱熹熾燃燈燎燐燒燔燕燗營燥燦燧燬燭燮燵燹燻燼燿爆爍爐爛爪爬爭爰爲爵父爺爻爼爽爾牀牆片版牋牌牒牘牙牛牝牟牡牢牧物牲牴特牽牾犀犁犂犇犒犠犢犧犬犯状犹狂狄狎狐狗狙狡狩狭狷狸狹狼狽猖猗猛猜猝猥猩猪猫献猴猶猷猾猿獄獅獎獗獣獨獪獰獲獵獸獺獻玄率玉王玖玩玲玻珀珂珈珊珍珞珠珪班現球琅理琉琢琥琲琳琴琵琶琺琿瑕瑙瑚瑜瑞瑟瑠瑣瑤瑪瑯瑰瑳瑶瑾璃璋璞璢環璽瓊瓏瓔瓜瓠瓢瓣瓦瓩瓮瓱瓲瓶瓷甌甍甎甑甕甘甚甜甞生産甥甦用甫田由甲申男甸町画畄畆畊界畏畑畔留畜畝畠畢畤略畦畧番畫異當畷畸畿疆疇疊疋疎疏疑疔疝疣疫疱疲疳疵疸疹疼疽疾痂病症痊痍痒痔痕痘痙痛痞痢痩痰痲痴痺痼痿瘁瘋瘍瘠瘡瘢瘤瘧瘰瘴瘻療癆癇癈癌癒癖癘癡癧癩癪癰癲癸発登發白百皃的皆皇皈皎皐皮皴皷皸皺皿盂盃盆盈益盍盖盗盛盜盞盟盡監盤盥盧盪目盲直相盻盾省眄眉看県眛眞真眠眤眦眩眷眸眺眼着睛睡督睥睦睨睹睾瞋瞎瞑瞞瞠瞥瞬瞭瞰瞳瞶瞹瞻瞼瞽瞿矇矛矜矢矣知矧矩短矮矯石砂砌砒研砥砦砧砲破硅硝硫硬硯硼碁碇碌碍碎碑碓碕碗碧碩碪確碼磁磅磊磐磔磚磧磨磬磯磴磽礁礎礑礙礦礪礫礬示礼社祀祁祇祈祐祓祕祖祗祚祝神祟祠祢祥票祭祷祿禀禁禄禅禊禍禎福禦禪禮禰禳禹禽禾禿秀私秉秋科秒秘租秡秣秤秦秧秩称移稀稈程稍税稔稗稚稜稟稠種稱稲稷稻稼稽稾稿穀穂穆穉積穏穗穡穢穩穫穰穴究穹空穽穿突窃窄窈窒窓窕窖窗窘窟窩窪窮窯窰窶窺竃竄竅竇竈竊立竏竒竓竕站竝竟章竡竢竣童竦竪竭端競竹竺竿笄笈笊笋笏笑笙笛笞笠笥符笨第笹筅筆筈等筋筌筍筏筐筑筒答策筥筧筬筮筱筵箆箇箋箏箒箔箕算箚箝箟管箪箭箱箴箸節篁範篇築篋篏篝篠篤篥篩篭篶簀簇簑簒簟簡簣簧簪簷簸簽簾簿籃籌籍籏籔籟籠籤籬米籾粁粂粃粉粍粒粕粗粘粛粟粢粥粧粮粱粲粳粹粽精糀糅糊糎糒糖糜糞糟糠糢糧糯糶糸糺系糾紀紂約紅紆紊紋納紐純紗紘紙級紛紜素紡索紫紬紮累細紲紳紹紺終絃組絅絆経絎結絖絛絞絡絢絣給絨絮統絲絶絹絽綏經継続綛綜綢綣綫綬維綮綰綱網綴綵綸綺綻綽綾綿緇緊緋総緑緒緘線緜緝緞締緡編緩緬緯練緻縁縄縅縊縋縒縛縞縟縡縢縣縦縫縮縱縷縹縺縻總績繁繃繆繊繋繍織繕繙繚繞繩繪繭繰繹繻繼繿纂纈續纎纏纒纓纔纖纜缸缺罅罌罍罎罐罔罕罘罨罩罪罫置罰署罵罷罸罹羃羅羈羊羌美羔羞羣群羨義羮羯羲羶羸羹羽翁翅翊翌習翔翠翩翫翰翳翹翻翼耀老考耄者耆耋而耐耒耕耗耘耙耜耡耨耳耶耻耽聆聊聖聘聚聞聟聨聯聰聲聳聴職聽聾肅肆肇肉肋肌肓肖肘肚肛肝股肢肥肩肪肭肯肱育肴肺胃胄胆背胎胖胙胚胛胝胞胡胤胥胯胱胴胸胼能脂脅脆脇脈脉脊脚脛脣脩脱脳脹脾腋腎腐腑腓腔腕腟腥腦腫腮腰腱腴腸腹腺腿膀膂膃膈膊膏膓膕膚膜膝膠膣膨膩膳膵膸膺膽膾膿臀臂臆臉臍臓臘臚臟臣臥臨自臭至致臺臻臼臾舁舂舅與興舉舊舌舍舎舐舒舖舗舘舛舜舞舟舩航般舳舵舶舷舸船艀艇艘艙艤艦艪艫艮良艱色艶艷艸艾芋芍芒芙芝芟芥芦芬芭花芳芸芹芻芽苅苑苒苔苗苛苜苞苟苣若苦苧苫英苴苹苺苻茂范茄茅茎茗茘茜茣茨茫茲茴茶茸茹荀荅草荊荏荐荒荘荳荷荻荼莇莊莖莚莞莟莠莢莨莫莱莵莽菁菅菊菌菓菖菘菜菟菠菩菫華菰菱菲菴菽萃萄萇萌萎萓萠萩萬萱萵萸萼落葆葉葎著葛葡葢董葦葩葬葯葱葵葺蒂蒄蒋蒐蒔蒙蒜蒟蒡蒭蒲蒸蒻蒼蒿蓁蓄蓆蓉蓊蓋蓍蓐蓑蓖蓙蓚蓬蓮蓴蓼蓿蔀蔑蔓蔔蔗蔚蔟蔡蔦蔬蔭蔵蔽蕁蕃蕈蕉蕊蕋蕎蕗蕘蕚蕣蕨蕩蕪蕭蕷蕾薀薄薇薈薐薑薔薗薙薜薦薨薩薪薫薬薯薹薺藁藉藍藏藐藜藝藤藥藩藪藷藹藺藻蘂蘆蘇蘊蘋蘓蘖蘗蘚蘭蘯蘿虎虐虔處虚虜虞號虫虱虹虻蚊蚓蚕蚤蚪蚫蚯蛆蛇蛉蛋蛔蛙蛛蛟蛤蛩蛭蛯蛸蛹蛻蛾蜀蜂蜃蜊蜍蜒蜘蜜蜥蜩蜴蜷蜻蜿蝋蝌蝎蝕蝗蝙蝟蝠蝦蝴蝶蝸蝿螂融螟螢螳螺螽蟀蟄蟆蟇蟋蟠蟯蟲蟹蟻蟾蠅蠑蠕蠢蠣蠧蠱蠶蠻血衂衆行衍衒術街衙衛衝衞衡衢衣表衫衰衲衷衽衾衿袁袂袈袋袍袒袖袙袞袢袤被袴袵袷裁裂装裏裔裕裘裙補裝裟裡裨裲裳裴裸裹製裾褄複褌褐褒褥褪褫褶褸褻襄襍襖襞襟襠襤襦襪襯襲襴襷西要覃覆覇覈覊見規覓視覗覘覚覦覧覩親覬覯覲観覺覽覿觀角觜解触觧觴觸言訂訃計訊討訐訓訖託記訛訝訟訣訥訪設許訳訴訶診註証詁詈詐詑詔評詛詞詠詢詣試詩詫詭詮詰話該詳詼誂誅誇誉誌認誑誓誕誘誚語誠誡誣誤誥誦誨説読誰課誹誼調諂諄談請諌諍諏諒論諛諜諡諤諦諧諫諭諮諱諳諷諸諺諾謀謁謂謄謌謎謐謗謙謚講謝謠謡謨謫謬謳謹譁證譌譎譏譖識譚譜警譫譬譯議譲譴護譽讀讃變讌讎讐讒讓讚谷谿豁豆豈豊豌豎豐豕豚象豢豪豫豸豹豺豼貂貅貌貎貝貞負財貢貧貨販貪貫責貭貮貯貰貳貴貶買貸費貼貽貿賀賁賂賃賄資賈賊賍賎賑賓賚賛賜賞賠賢賣賤賦質賭賺賻購賽贄贅贈贊贋贍贏贓贔贖赤赦赧赫赭走赱赴起趁超越趙趣趨足趺趾跂跋跌跏跛距跟跡跣跨跪路跳踈踊踏踐踝踞踪踰踵蹂蹄蹈蹉蹊蹌蹕蹙蹟蹠蹤蹲蹴蹶躁躄躅躇躊躋躍躓躙躪身躬躯躰躱躾軆軈車軋軌軍軒軟転軫軸軻軼軽軾較載輊輌輒輓輔輕輙輛輜輝輟輦輩輪輯輳輸輻輾輿轂轄轅轆轉轍轎轜轟轡轢轣轤辛辜辞辟辧辨辭辯辰辱農辷辺辻込辿迂迄迅迎近返迚迥迦迩迫迭迯述迴迷迸迹追退送逃逅逆逋逍透逐逑逓途逕逗這通逝逞速造逡逢連逮週進逵逶逸逹逼逾遁遂遅遇遊運遍過遏遐遑道達違遖遙遜遞遠遡遣遥適遭遮遯遲遵遶遷選遺遼遽避邀邁邂邃還邇邉邊邏邑那邦邨邪邯邱邵邸郁郊郎郡郤部郭郵郷都鄙鄭鄰鄲酉酊酋酌配酎酒酔酖酢酣酥酩酪酬酵酷酸醂醇醉醋醍醐醒醜醢醤醪醫醴醵醸醺釀釆采釈釉釋里重野量釐金釘釜針釣釦釧釵釼鈍鈎鈑鈔鈕鈞鈴鈷鈿鉄鉅鉈鉗鉚鉛鉞鉢鉤鉦鉱鉾銀銃銅銑銓銕銖銘銚銛銜銭銷銹鋏鋒鋤鋩鋪鋭鋲鋸鋼錆錐錘錙錚錠錢錣錦錨錫錬錮錯録錺錻鍄鍋鍍鍔鍛鍜鍬鍮鍵鍼鍾鎌鎔鎖鎗鎚鎧鎬鎭鎮鏃鏈鏑鏖鏝鏡鏤鐃鐐鐔鐘鐙鐡鐫鐵鐶鐸鐺鑄鑑鑒鑓鑚鑛鑞鑢鑪鑰鑵鑷鑼鑽鑾鑿钁長門閃閇閉閊開閏閑間閔閘閙関閣閤閥閧閨閭閲閻閼闇闊闌闍闔闕闖闘關闡闢闥阜阡阪阮阯防阻阿陀陂附陋陌降限陛陞陟院陣除陥陪陬陰陳陵陶陷陸険陽隅隆隈隊隋隍階随隔隕隘隙際障隠隣隧隨險隰隱隴隷隸隹隻隼雀雁雄雅集雇雉雌雍雑雕雖雙雛雜離難雨雪雫雰雲零雷雹電需霄霆震霊霍霎霏霑霓霖霜霞霧霰露霸霹霽霾靂靄靆靈靉青靖静靜非靠靡面革靭靱靴靼鞄鞅鞆鞋鞍鞏鞘鞠鞣鞫鞭鞴韃韋韓韜韮韲音韵韶韻響頁頂頃項順須頌頏預頑頒頓頗領頚頡頤頬頭頴頷頸頻頼頽顆題額顎顏顔顕願顛類顧顫顯顰顱顳風颪颯颱颶飃飄飛飜食飢飩飫飭飮飯飲飴飼飽飾餅餉養餌餐餒餓餔餘餝餞館餬餾饂饅饉饋饌饑饒饗首馗香馥馨馬馭馮馳馴駁駄駅駆駈駐駑駒駕駘駛駝駢駭駱駸駿騁騎騏騒験騙騨騰騷騾驀驅驕驗驚驛驟驢驩驪骨骰骸骼髀髄髓體高髣髪髭髮髯髴髷髻鬆鬘鬚鬟鬢鬣鬧鬨鬪鬮鬱鬲鬻鬼魁魂魃魄魅魏魔魚魯魴鮃鮎鮑鮒鮓鮨鮪鮫鮭鮮鮹鯉鯑鯔鯖鯛鯢鯣鯤鯨鯰鯵鰊鰌鰍鰐鰒鰓鰕鰛鰡鰤鰥鰭鰮鰯鰹鰺鰻鱈鱒鱗鱶鱸鳥鳧鳩鳫鳰鳳鳴鳶鴆鴇鴈鴉鴒鴛鴟鴦鴨鴻鴿鵄鵑鵙鵜鵝鵞鵠鵤鵬鵯鵲鵺鶉鶏鶯鶴鶺鷄鷲鷹鷺鸚鸞鹵鹸鹹鹽鹿麁麈麋麑麒麓麗麝麟麥麦麩麪麭麹麺麻麼麾麿黄黌黍黎黏黒默黙黛黜黝點黠黨黯黴黽鼇鼈鼎鼓鼠鼬鼻鼾齊齋齎齒齟齡齢齣齦齧齪齬齶齷龍龕龜Ｂｃｏ～𪜈'
	vocab=sorted(set(vocab))
	char_to_int={j:i for i,j in enumerate((vocab))}
	int_to_char={i:j for i,j in enumerate((vocab))}
	return metadata,vocab,char_to_int,int_to_char

class custom_dataset(data.Dataset):
    def __init__(self, data, scale=1, length=1024):
        super(custom_dataset, self).__init__()
        self.data = data
        self.scale = scale
        self.length = length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = Image.open(self.data[index][0]).convert('RGB')
        vertices = np.array(self.data[index][2])    
        labels=np.array([1]*len(self.data[index][2]))
#         img, vertices = adjust_height(img, vertices) 
#         img, vertices = rotate_img(img, vertices)
#         img, vertices = crop_img(img, vertices, labels, self.length) 
        img,vertices=resize_img_and_vertices(img,vertices,self.length)
        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])

        score_map, geo_map, ignored_map = get_score_geo(img, vertices, labels, self.scale, self.length)
        return transform(img), score_map, geo_map, ignored_map