# -*- coding: utf-8 -*-
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
import os
import cv2
import matplotlib.pyplot as plt
import random
import math
from skimage import data,filters, segmentation, measure,morphology, color
import cv



def load_data(file_path):
    f = open(file_path, 'rb')  # 二进制打开
    data = []
    img = image.open(f)  # 以列表形式返回图片像素值
    m, n = img.size  # 活的图片大小
    for i in range(m):
        for j in range(n):  # 将每个像素点RGB颜色处理到0-1范围内并存放data
            x, y, z = img.getpixel((i, j))
            data.append([x / 255.0, y / 255.0, z / 255.0])
    f.close()
    return np.mat(data), m, n  # 以矩阵型式返回data，图片大小


def sub_file(file_dir):
    lst = os.listdir(file_dir)
    lst_detail = []
    for i in range(len(lst)):
        lst_detail.append(file_dir + lst[i])
    return lst_detail, lst
        

def random_sample_cluster(file_dir):
    lst_file,_ = sub_file(file_dir)
    random_num = range(len(lst_file))
    random.shuffle(random_num)
    data = []
    for i in range(20):
        file_name = lst_file[random_num[i]]
        f = open(file_name, 'rb')
        img = image.open(f)
        m, n = img.size
        for j in range(m):
            for k in range(n):
                x, y, z = img.getpixel((j, k))
                data.append([x / 255.0, y / 255.0, z / 255.0])
    print 'start cluster'
    kmeans = KMeans(n_clusters=6, random_state=0).fit(data)  
    np.save('center.npy', kmeans.cluster_centers_)
    return kmeans.cluster_centers_             

#膨胀
def Expand(image):
    w = image.shape[1]#colum
    h = image.shape[0]#row
#    size = (w,h)
    iExpand = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            iExpand[i][j] = 255
    for i in range(h):
        for j in range(w):
            if image[i][j] == 0:
                for k in range(5):
                    for l in range(5):
                        if -1<(i-2+k)<h and -1<(j-2+l)<w:
                            iExpand[i-2+k,j-2+l] = 0
    return iExpand  

def lablize_visualize(clu_center, file_dir):
    lst_file, file_lst = sub_file(file_dir)
    for file_index in range(len(lst_file)):
        f = open(lst_file[file_index], 'rb')
        img = image.open(f)
        m, n = img.size
        label_img = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                x, y, z = img.getpixel((i, j))
                minnum = 1000
                for index in range(len(clu_center)):              
                    diff = math.pow(x / 255.0 - clu_center[index][0], 2) + \
                           math.pow(y / 255.0 - clu_center[index][1], 2) + \
                           math.pow(z / 255.0 - clu_center[index][2], 2)
                    if minnum > diff:
                        minnum = diff
                        label_img[i][j] = index
        #save result image
        gray = image.new('L', (m, n))               
        for i_row in range(m):
            for j_col in range(n):
                if label_img[i_row][j_col] != 2:
                    img.putpixel((i_row, j_col), (255, 255, 255))
                    gray.putpixel((i_row, j_col), 255)
                else:
                    gray.putpixel((i_row, j_col), 0)
        img.save('./channel3result/' + file_lst[file_index])   
        #img.show()
        
        temp = np.asarray(gray)
        #开运算
        temp = morphology.opening(temp, morphology.disk(3))
        temp.flags.writeable = True 
        #连通域处理
        lst_propoerty = []
        lst_index = []
        labeled_img, nums = measure.label(temp, return_num=True, background=255, connectivity=2)
        con_domain_fields = measure.regionprops(labeled_img)
        for i in range(len(con_domain_fields)):
            if con_domain_fields[i].area >= 500 or con_domain_fields[i].area <=20:
                for index_var in con_domain_fields[i].coords:
                    temp[index_var[0]][index_var[1]] = 255
            else:
                lst = []
                lst.append(con_domain_fields[i].area)
                lst.append(con_domain_fields[i].equivalent_diameter)
                lst.append(con_domain_fields[i].perimeter)
                lst_propoerty.append(lst)
                lst_index.append(i)
        im = image.fromarray(temp)
        im.save('./con_domain_fields/' + file_lst[file_index])
        #im.show()  
        #细菌分类聚类                 
                lst.append(con_domain_fields[i].area)
                lst.append(con_domain_fields[i].equivalent_diameter)
                lst.append(con_domain_fields[i].perimeter)
                lst_propoerty.append(lst)
                lst_index.append(i)
        im = image.fromarray(temp)
        im.save('./con_domain_fields/' + file_lst[file_index])
        #im.show()  
        #细菌分类聚类                 
        kmeans = KMeans(n_c
        kmeans = KMeans(n_clusters=2, random_state=0).fit(lst_propoerty)           
        la = kmeans.labels_       
        
        for cor_index in range(len(lst_index)):
            for var in con_domain_fields[lst_index[cor_index]].coords:
                if la[cor_index] == 0:
                    temp[var[0]][var[1]] = 10
                else:
                    temp[var[0]][var[1]] = 200

        ul_img = image.fromarray(temp)
        ul_img.save('./ulimg/' + file_lst[file_index])    

        
        #a = raw_input()
        #im.save('./expand/' + file_lst[file_index])        
        
        
        
#        im = image.fromarray(temp)
#        
#        im.save('./con_domain_fields/' + file_lst[file_index])
        #im.show()
       
      
        #闭运算
                lst.append(con_domain_fields[i].area)
                lst.append(con_domain_fields[i].equivalent_diameter)
                lst.append(con_domain_fields[i].perimeter)
                lst_propoerty.append(lst)
                lst_index.append(i)
        im = image.fromarray(temp)
        im.save('./con_domain_fields/' + file_lst[file_index])
        #im.show()  
        #细菌分类聚类                 
        kmeans = KMeans(n_c
#        gray.show('gray')        
#        
#        bw = morphology.closing(temp, morphology.square(3)) 
#                
#        gray =  image.fromarray(bw)
#        gray.show('biyunsuan')
        #连通域
        
        
#        #第二次聚类
#        data2 = []
#        for i in range(m):
#            for j in range(n):  # 将每个像素点RGB颜色处理到0-1范围内并存放data
#                x, y, z = img.getpixel((i, j))
#                data2.append([x / 255.0, y / 255.0, z / 255.0])
#        label = KMeans(n_clusters = 3).fit_predict(data2)
#        label = label.reshape([m, n])
#        for index_row in range(m):
#            for index_col in range(n):
#                if label[index_row][index_col] != 2:
#                    img.putpixel((index_row, index_col), (255,255,255))
#        img.save('./clu2/' + file_lst[file_index])            
#        
#        
#        #二次聚类后的二值化图像
#        gray_pic = image.new("L", (m, n))
#        print 'row:' + str(m)
#        print 'col:' + str(n)
#
#        for index_row in range(m):
#            for index_col in range(n):
#                if label[index_row][index_col] != 2:
#                    gray_pic.putpixel((index_row, index_col), 255)
#                else:
#                    gray_pic.putpixel((index_row, index_col), 0)
#
#        labeled_img, nums = measure.label(np.asarray(gray_pic), return_num=True)
#        con_domain_fields = measure.regionprops(labeled_img)
#        for i in range(len(con_domain_fields)):
#            if con_domain_fields[i].area >= 5:
#                for index_var in con_domain_fields[i].coords:
#                    gray_pic.putpixel((index_var[1], index_var[0]), 255)
#                else:
#                    gray_pic.putpixel((index_var[1], index_var[0]), 0)
#        gray_pic.save('./con_domain_fields/' + file_lst[file_index])


if __name__ == '__main__':
#    file_dir = './12/'
#    file_name = os.lisrandom_sample_clustertdir(file_dir)
#    for var in file_name:
#        temp = file_dir + var
#        img, row, col = load_data(temp)
#        label = KMeans(n_clusters=4).fit_predict(img)  # 聚类中心的个数为4
#        label = label.reshape([row, col])  # 聚类获得每个像素所属的类别
#        pic_new = image.new("L", (row, col))  # 创建一张新的灰度图保存聚类后的结果
#
#        for i in range(row):  # 根据所属类别向z图片中添加灰度值
#            for j in range(col):
#                pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
#
#
#        pic_new.save('./result/'+var)
#        pic_new = np.asarray(pic_new)
#
#        ret, thresh1 = cv2.threshold(pic_new, 127, 255, cv2.THRESH_BINARY)
#
#        plt.imshow(thresh1)
#        plt.show()
    #print 'start random cluster'
    #centers = random_sample_cluster('./sample/')
    
    centers = np.load('center.npy')  
    print 'start labelize image'
    gray = lablize_visualize(centers, './sample/')
    a = np.asarray(gray)



