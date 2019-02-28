import warnings
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras.backend as K
K.set_image_dim_ordering('th')
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dropout, UpSampling2D, ZeroPadding2D
from keras.layers import Permute, Reshape, Activation
from keras.utils.data_utils import get_file
from datasets import CONFIG


# CITYSCAPES MODEL
def get_dilation_model_cityscapes(input_shape, classes):
    # 构建模型
    model_in = Input(shape=input_shape)    
    h = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(model_in)  # (,64,1394,1394)
    h = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(h)         # (,64,1392,1392)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)  # (,64,696,696)
    h = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(h)        # (,128,694,694)
    h = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(h)        # (,128,692,692)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)  # (,128,346,346)
    h = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(h)        # (,256,344,344)
    h = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(h)        # (,256,342,342)
    h = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(h)        # (,256,340,340)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)  # (,256,170,170)
    h = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(h)        # (,512,168,168)
    h = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(h)        # (,512,166,166)
    h = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(h)        # (,512,164,164)
    h = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_1')(h)  # (,512,160,160) 扩张卷积比普通卷积对于尺度的缩小作用更大
    h = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2')(h)  # (,512,156,156) 扩张卷积
    h = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3')(h)  # (,512,152,152) 扩展卷积
    h = Conv2D(4096,(7, 7), dilation_rate=(4, 4), activation='relu', name='fc6')(h)     # (,4096,128,128)  扩张后的卷积核大小为 25*25
    h = Dropout(0.5, name='drop6')(h)
    h = Conv2D(4096, (1, 1), activation='relu', name='fc7')(h)   # (,4096,128,128)
    h = Dropout(0.5, name='drop7')(h)
    h = Conv2D(classes, (1, 1), name='final')(h)                   # (,19,128,128)  classes=19 最后要在通道上取softmax，因此通道数等于类别数
    h = Conv2D(classes, (3, 3), padding='same', activation='relu', name='ctx_conv1_1')(h)  # (,19,128,128)
    h = Conv2D(classes, (3, 3), padding='same', activation='relu', name='ctx_conv1_2')(h)  # (,19,128,128)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = Conv2D(classes, (3, 3), dilation_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)   # (,19,128,128)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = Conv2D(classes, (3, 3), dilation_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)   # (,19,128,128)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = Conv2D(classes, (3, 3), dilation_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)   # (,19,128,128)
    h = ZeroPadding2D(padding=(16, 16))(h)
    h = Conv2D(classes, (3, 3), dilation_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)  # (,19,128,128)
    h = ZeroPadding2D(padding=(32, 32))(h)
    h = Conv2D(classes, (3, 3), dilation_rate=(32, 32), activation='relu', name='ctx_conv6_1')(h)  # (,19,128,128)
    h = ZeroPadding2D(padding=(64, 64))(h)
    h = Conv2D(classes, (3, 3), dilation_rate=(64, 64), activation='relu', name='ctx_conv7_1')(h)  # (,19,128,128)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(classes, (3, 3), activation='relu', name='ctx_fc1')(h)     # (,19,128,128)
    h = Conv2D(classes, (1, 1), name='ctx_final')(h)                      # (,19,128,128)

    # the following two layers pretend to be a Deconvolution with grouping layer.
    # never managed to implement it in Keras
    # since it's just a gaussian upsampling trainable=False is recommended
    # tensorflow实现中用 tf.image.resize_bilinear 执行双线性插值上采样
    h = UpSampling2D(size=(8, 8))(h)        # 上采样8倍 (,19,1024,1024)  
    # 使用卷积操作构造一个高斯上采样，这里trainable=False
    logits = Conv2D(classes, (16, 16), padding='same', use_bias=False, trainable=False, name='ctx_upsample')(h)  # (,19,1024,1024)

    # 在通道层面施加softmax
    _, c, h, w = logits._keras_shape            # (None,19,1024,1024)
    x = Permute(dims=(2, 3, 1))(logits)         # (None,1024,2024,19)
    x = Reshape(target_shape=(h * w, c))(x)     # (None,1048576,19)
    x = Activation('softmax')(x)                # (None,1048576,19)
    x = Reshape(target_shape=(h, w, c))(x)      # (None,1024,1024,19)
    model_out = Permute(dims=(3, 1, 2))(x)      # (None,19,1024,1024)
    
    # 构建模型
    model = Model(input=model_in, output=model_out, name='dilation_cityscapes')
    return model


# model function
def DilationNet(dataset, pretrained=True):
    """ 根据有关参数，初始化模型，导入权重
    """
    classes = CONFIG[dataset]['classes']          # 类别数目
    input_shape = CONFIG[dataset]['input_shape']  # 采用th顺序，维度在前

    # get the model
    if dataset == 'cityscapes':
        model = get_dilation_model_cityscapes(input_shape=input_shape, 
                                              classes=classes)
    # 导入权重
    if pretrained:
        assert K.image_dim_ordering() == 'th'
        weights_path = get_file("cityscapes.h5", origin=None, 
            cache_subdir='models') 
        model.load_weights(weights_path)

    return model


if __name__ == '__main__':
    ds = 'cityscapes'  # choose between cityscapes, kitti, camvid, voc12
    # get the model
    model = DilationNet(dataset=ds)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.summary()
