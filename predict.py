# -*- coding: utf-8 -*-

import numpy as np
import cv2
from dilation_net import DilationNet
from datasets import CONFIG
import matplotlib.pyplot as plt


# predict function, mostly reported as it was in the original repo
def predict(image, model, ds):
    image = image.astype(np.float32) - CONFIG[ds]['mean_pixel']  # 均值处理
    conv_margin = CONFIG[ds]['conv_margin']  # 输入1396,输出1024. (1396-1024)/2=186

    input_dims = (1,) + CONFIG[ds]['input_shape']   # (1, 3, 1396, 1396),
    batch_size, num_channels, input_height, input_width = input_dims
    model_in = np.zeros(input_dims, dtype=np.float32)

    image_size = image.shape                         # (1024, 2048, 3)
    output_height = input_height - 2 * conv_margin   # 1024
    output_width = input_width - 2 * conv_margin     # 1024
    # 扩充边缘，使图像变大. 大小转为(1396,2420). 便于在经过model前向传播后还原为(1024,2048)
    image = cv2.copyMakeBorder(image, conv_margin, conv_margin, 
                               conv_margin, conv_margin,
                               cv2.BORDER_REFLECT_101)
    # 此处image_size可以整除output_height，后一项为0.
    # 前一项将图像划分为多个模块. num_tiles_h=1, num_tiles_w=2
    num_tiles_h = image_size[0] // output_height + (1 if image_size[0] % output_height else 0)
    num_tiles_w = image_size[1] // output_width + (1 if image_size[1] % output_width else 0)

    row_prediction = []
    # 按照行、列分别循环进行预测，最后将预测结果合并起来.
    for h in range(num_tiles_h):
        col_prediction = []
        for w in range(num_tiles_w):
            # 根据 h,w 提取测试图像中的对应区域 tile.shape=(1396,1396,3)
            offset = [output_height * h, output_width * w]
            tile = image[offset[0]:offset[0] + input_height,
                         offset[1]:offset[1] + input_width, :]
            # 将切分出来的图像加上边框变为 margin=(0,0,0,0) tile.shape=(1396,1396,3)
            margin = [0, input_height - tile.shape[0],
                      0, input_width - tile.shape[1]]
            tile = cv2.copyMakeBorder(tile, margin[0], margin[1],
                                      margin[2], margin[3],
                                      cv2.BORDER_REFLECT_101)
            model_in[0] = tile.transpose([2, 0, 1])

            # 预测结果
            prob = model.predict(model_in)[0]       # (19,1024,1024)
            col_prediction.append(prob)

        col_prediction = np.concatenate(col_prediction, axis=2)  
        row_prediction.append(col_prediction)
    prob = np.concatenate(row_prediction, axis=1)   # (19,1024,2048)
    # 取 argmax，获取颜色
    prediction = np.argmax(prob, axis=0)
    color_image = CONFIG[ds]['palette'][prediction.ravel()].reshape(image_size)

    return color_image


if __name__ == '__main__':

    ds = 'cityscapes'  # choose between cityscapes, kitti, camvid, voc12
    model = None
    # get the model
    model = DilationNet(dataset=ds)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.summary()

    # read and predict a image
    fname = "imgs_test/fig4.png"
    im = cv2.imread(fname)
    y_img = predict(im, model, ds)

    # plot results
    fig = plt.figure(figsize=(10,5))
    a = fig.add_subplot(1, 2, 1)   # 一行两列，分别显示原图和分割后的图片
    imgplot = plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    a.set_title('Image')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(y_img)
    a.set_title('Semantic segmentation')
    plt.show()

