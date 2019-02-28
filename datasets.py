import numpy as np
# 对原 Github 中的内容进行了删减
# configuration for different datasets
CONFIG = {
    'cityscapes': {
        'classes': 19,
        'weights_file': 'weight/cityscapes.h5',
        'weights_url': 'http://imagelab.ing.unimore.it/files/dilation_keras/cityscapes.h5',
        'input_shape': (3, 1396, 1396),
        'test_image': ['imgs_test/fig1.png','imgs_test/fig2.png','imgs_test/fig3.png','imgs_test/fig4.png'], 
        'mean_pixel': (72.39, 82.91, 73.16),
        'palette': np.array([[128, 64, 128],
                            [244, 35, 232],
                            [70, 70, 70],
                            [102, 102, 156],
                            [190, 153, 153],
                            [153, 153, 153],
                            [250, 170, 30],
                            [220, 220, 0],
                            [107, 142, 35],
                            [152, 251, 152],
                            [70, 130, 180],
                            [220, 20, 60],
                            [255, 0, 0],
                            [0, 0, 142],
                            [0, 0, 70],
                            [0, 60, 100],
                            [0, 80, 100],
                            [0, 0, 230],
                            [119, 11, 32]], dtype='uint8'),
        'zoom': 1,
        'conv_margin': 186   # 输入1396,输出1024. (1396-1024)/2=186
    }
}
