from cv2 import cv2
import glob
import numpy as np

picture_threshold=10 #阈值数值


def picture_line_detect(path):#图像处理功能
    new_image = cv2.imread(path)
    new_image = new_image[20:100, ]
    # cv2.imshow("new_image", new_image)###########
    new_gray = new_image.copy()   #保证原始图像不做改变
    new_gray = cv2.cvtColor(new_gray,cv2.COLOR_BGR2GRAY)
    ##############图像sobel算子#################
    new_gray = cv2.GaussianBlur(new_gray, (3, 3), 6)#高斯处理
    cv2.imshow("new_gray", new_gray)

    kernel1 = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],), dtype="float32")
    #-1表示输入输出维度相同
    dst1 = cv2.filter2D(new_gray, -1, kernel1)

    # histimg = cv2.equalizeHist(dst1)
    # cv2.imshow("histimg",histimg)

    # keral1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # mediablur=cv2.morphologyEx(dst2,cv2.MORPH_OPEN,keral1,iterations=1)
    # cv2.imshow("11img_erode",mediablur)

    R, dst1 = cv2.threshold(dst1, 10, 255, cv2.THRESH_BINARY)
    drawimg1 = np.zeros((dst1.shape[0], dst1.shape[1], 1), np.uint8)
    #dst1:图像    cv2.RETR_TREE:建立一个等级树结构的轮廓（4种轮廓的检索模式之一）
    #cv2.CHAIN_APPROX_SIMPLE 压缩水平方向、垂直方向、对角线方向的元素，只保留该方向的终点坐标（矩形只需四顶点） 三种轮廓的近似办法之一
    contours, sfas_ = cv2.findContours(dst1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(drawimg,contours,-1,(0,0,255),1)
    #cv2.imshow("dst1", dst1)

    if contours:
        for i in range(len(contours)):
            rect = cv2.boundingRect(contours[i])  # 得到目标区域的中心点和旋转角度
            #print(i, ":", rect)
            if  (rect[3] > 40):
                drawimg2 = cv2.drawContours(drawimg1, [contours[i]], -1, (255), cv2.FILLED)

    #cv2.imshow("drawimg2", drawimg2)

    # drawimg3 =  np.zeros((drawimg2.shape[0],drawimg2.shape[1],1),np.uint8)
    drawimg10 = drawimg2
    #cv2.imshow("drawimg10", drawimg10)

    contours, _ = cv2.findContours(drawimg10, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(drawimg,contours,-1,(0,0,255),1)

    max_rect = 0
    max_flage = 0
    if contours:
        for i in range(len(contours)):
            rect = cv2.boundingRect(contours[i])  # 得到目标区域的中心点和旋转角度

            if rect[3] > 11:
                drawing3 = cv2.drawContours(drawimg10, [contours[i]], -1, (255), cv2.FILLED)

    keral1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    drawimg4 = cv2.morphologyEx(drawimg10, cv2.MORPH_CLOSE, keral1, iterations=3)
    #cv2.imshow("drawimg4", drawimg4)
    '''
    drawimg3 =  np.zeros((dst1.shape[0],dst1.shape[1],1),np.uint8)
    contours,sfas_ = cv2.findContours(drawimg4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(drawimg,contours,-1,(0,0,255),1)

    if contours:
        for i in range(len(contours)):
            rect = cv2.boundingRect(contours[i])  #得到目标区域的中心点和旋转角度
            print(i,":",rect)
            if rect[3] > 40:
                cv2.drawContours(drawimg3,[contours[i]],-1,(255),cv2.FILLED)
    cv2.imshow("drawimg5",drawimg3)
    '''

    # drawimg3=np.squeeze(drawimg3,axis=2)
    # print(drawimg3.shape)
    # np.savetxt('sdf.csv',drawimg3)

    # canny_img = cv2.Canny(drawimg3,100,120,0)

    lines = cv2.HoughLines(drawimg10, 1, np.pi / 180 * 2, 35)
    result = np.zeros((drawimg10.shape[0], drawimg10.shape[1], 1), np.uint8)
    for line in lines:
        rho = line[0][0]  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta

        if (theta < 0.5) or (theta > 3):  # 垂直直线

            pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
            # 该直线与最后一行的焦点
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
            cv2.line(result, pt1, pt2, (255, 255, 255))  # 绘制一条白线
        '''elif theta > 87 and theta < 95:                                                  #水平直线
            pt1 = (0,int(rho/np.sin(theta)))               # 该直线与第一列的交点
            #该直线与最后一列的交点
            pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
            cv2.line(result, pt1, pt2, (255,255,255), 1)           # 绘制一条直线
            '''
    #cv2.imshow("result", result)

    result_shuchu=result[45]
    result_shuchu=np.squeeze(result_shuchu,-1)
    return(result_shuchu)
if __name__ == '__main__':
    picture_path=glob.glob('D:/Challenge Cup/wwl/xin/broken_line/*.png')
    for picture_count in range(len(picture_path)):
        print(picture_count)
        test=picture_line_detect(picture_path[picture_count])
        jisuan = int(0)
        for i in range(len(test)):
            if test[i] >= 200:
                # print(i)
                if i - jisuan > 150:
                    print(f'发现断纱在x={ (i - jisuan) / 2},在第{picture_count}张图')
                jisuan = i





