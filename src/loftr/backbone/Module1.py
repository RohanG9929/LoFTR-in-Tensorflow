import tensorflow as tf
from einops.einops import rearrange


def Module1(x):
    x = rearrange(x, 'n c h w -> n h w c')

    y=tf.keras.layers.Conv2D(128,7,strides=2,padding='same',use_bias=False)(x) # (no of kernels,kernel size,stride,padding,activation,bias)
    z1=tf.keras.layers.BatchNormalization(axis=(1,2))(y)
    out1=tf.keras.layers.ReLU()(z1)
    # print("the shape after the first building block is")
    # print(out1.shape)

    # the three layer structure
    # layer 1
    out_layer1=tf.keras.layers.Conv2D(128,3,strides=1,padding='same',use_bias=False)(out1) # first argument is filters which is the number of filters in the output space
    #out_layernorm1=tf.keras.layers.BatchNormalization(axis=(1,2))(out_layer1)
    out_layernorm1=tf.keras.layers.BatchNormalization(axis=-1)(out_layer1)
    layer1_relu=tf.keras.layers.ReLU()(out_layernorm1)


    out_layer1_agreg=tf.keras.layers.Conv2D(128,3,strides=1,padding='same',use_bias=False)(layer1_relu)
    #out_layer1_agreg_normalized=tf.keras.layers.BatchNormalization(axis=(1,2))(out_layer1_agreg)
    out_layer1_agreg_normalized=tf.keras.layers.BatchNormalization(axis=-1)(out_layer1_agreg)
    layer_one_return=tf.keras.layers.ReLU()(out1+out_layer1_agreg_normalized)
    # print('The shape after layer one return is')
    # print(layer_one_return.shape) # this is 1/2 of the original image


    # layer 2
    # downsampling
    x=tf.keras.layers.Conv2D(196,1,strides=2,padding='valid',use_bias=False)(layer_one_return)
    x_downsampled=tf.keras.layers.BatchNormalization(axis=-1)(x)
    # print(f'the shape of x_downsampled is {x_downsampled.shape}')

    # normal layer2 shape is (batch,height,width,number of filters)
    out_layer2=tf.keras.layers.Conv2D(196,3,strides=2,padding='valid',use_bias=False)(layer_one_return) # first argument is the initial dimensions.
    # print(f'output 2 without tf.pad is {out_layer2.shape}')
    paddings_2_1 = tf.constant([[0,0],[1, 0], [1, 0],[0,0]])
    out_layer2_trial=tf.pad(out_layer2,paddings_2_1)
    # print(f'the shape of output layer using tf.pad is{out_layer2_trial.shape}')
    out_layernorm2=tf.keras.layers.BatchNormalization(axis=-1)(out_layer2_trial) 
    layer2_relu=tf.keras.layers.ReLU()(out_layernorm2)

    out_layer2_agreg=tf.keras.layers.Conv2D(196,3,strides=1,padding='valid',use_bias=False)(layer2_relu)
    # print(f'shape before implementing second tf.pad is{out_layer2_agreg.shape}')
    paddings_2_2 = tf.constant([[0,0],[1, 1], [1, 1],[0,0]])
    out_layer2_trial_2=tf.pad(out_layer2_agreg,paddings_2_2)
    # print(f'the shape of output layer using tf.pad is{out_layer2_trial_2.shape}')
    out_layer2_agreg_normalized=tf.keras.layers.BatchNormalization(axis=-1)(out_layer2_trial)
    layer_two_return=tf.keras.layers.ReLU()(out_layer2_agreg_normalized+x_downsampled)
    # print('the shape of layer 2 after implementing tf.pad')
    # print(layer_two_return.shape) # this is 1/4 of the original image

    # layer 3 downsampling
    x_new=tf.keras.layers.Conv2D(256,1,strides=2,padding='valid',use_bias=False)(layer_two_return)
    x_downsampled_new=tf.keras.layers.BatchNormalization(axis=-1)(x_new)
    # print(f'the shape of x_downsampled is {x_downsampled_new.shape}')
    # layer 3
    out_layer3=tf.keras.layers.Conv2D(256,3,strides=2,padding='valid',use_bias=False)(layer_two_return) # first argument is the initial dimensions.
    paddings_3_1 = tf.constant([[0,0],[1, 0], [1, 0],[0,0]])
    out_layer3_trial=tf.pad(out_layer3,paddings_3_1)
    # print(f'the shape of third output layer using tf.pad is{out_layer3_trial.shape}')

    out_layernorm3=tf.keras.layers.BatchNormalization(axis=-1)(out_layer3_trial)
    layer3_relu=tf.keras.layers.ReLU()(out_layernorm3)


    # aggregated layer 3
    out_layer3_agreg=tf.keras.layers.Conv2D(256,3,strides=1,padding='valid',use_bias=False)(layer3_relu)
    paddings_3_2 = tf.constant([[0,0],[1, 1], [1, 1],[0,0]])
    out_layer3_trial_3=tf.pad(out_layer3_agreg,paddings_3_2)
    # print(f'the shape of output layer in line 73 using tf.pad is{out_layer3_trial_3.shape}')
    out_layer3_agreg_normalized=tf.keras.layers.BatchNormalization(axis=-1)(out_layer3_trial)
    layer_three_return=tf.keras.layers.ReLU()(out_layer3_agreg_normalized+x_downsampled_new)
    # print('the shape after layer three')
    # print(layer_three_return.shape) # this is 1/8 of the original image

    # FPN
    # layer 1
    outLayer3_outconv=tf.keras.layers.Conv2D(256,1,strides=1,padding='same',use_bias=False)(layer_three_return)  # shape is (1,80,60,265)
    # print(f'x3 out shape is{outLayer3_outconv.shape}')
    outLayer3_outconv_downsampled=tf.image.resize(outLayer3_outconv,[int(outLayer3_outconv.shape[1]*2),int(outLayer3_outconv.shape[2]*2)]) # shape is (1,40,30,265)
    # print(f'shape after first interpolate is {outLayer3_outconv_downsampled.shape}')
    outLayer2_outconv=tf.keras.layers.Conv2D(256,1,strides=1,padding='same',use_bias=False)(layer_two_return)
    #outLayer2_outconv_downsampled=tf.image.resize(outLayer2_outconv,[int(outLayer2_outconv.shape[1]*0.5),int(outLayer2_outconv.shape[2]*0.5)]) # shape is (1,20,15,265)

    # now the sequential model layer2_outconv2
    model_outlayer2_outconv=tf.keras.Sequential()
    model_outlayer2_outconv.add(tf.keras.layers.Conv2D(256,3,strides=1,padding='same',use_bias=False))
    model_outlayer2_outconv.add(tf.keras.layers.BatchNormalization(axis=-1))
    model_outlayer2_outconv.add(tf.keras.layers.LeakyReLU())
    model_outlayer2_outconv.add(tf.keras.layers.Conv2D(196,3,strides=1,padding='same',use_bias=False))

    x2_out=model_outlayer2_outconv(outLayer3_outconv_downsampled+outLayer2_outconv)
    # print('shape after first final addition')
    # print(x2_out.shape)

    # the issue is with downsampling of tensors.
    # there is an issue in the code when the stride is equal to 1
    x2_out_2x=tf.image.resize(x2_out,[int(x2_out.shape[1]*2),int(x2_out.shape[2]*2)])
    # print('the shape after the second downsample')
    # print(x2_out_2x.shape)
    x1_out=tf.keras.layers.Conv2D(196,1,strides=1,padding='same',use_bias=False)(layer_one_return)
    # print(f'the shape after x1 out is{x1_out.shape}')

    model_outlayer1_outconv=tf.keras.Sequential()
    model_outlayer1_outconv.add(tf.keras.layers.Conv2D(196,3,strides=1,padding='same',use_bias=False))
    model_outlayer1_outconv.add(tf.keras.layers.BatchNormalization(axis=-1))
    model_outlayer1_outconv.add(tf.keras.layers.LeakyReLU())
    model_outlayer1_outconv.add(tf.keras.layers.Conv2D(128,3,strides=1,padding='same',use_bias=False))
    x1_out=tf.keras.layers.Conv2D(196,1,strides=1,padding='same',use_bias=False)(layer_one_return)
    # print(x1_out.shape)

    # print('this is the final')
    x1_out=model_outlayer1_outconv(x1_out+x2_out_2x)

    # print(f'x3 shape is {outLayer3_outconv.shape}')
    # print(f'x1 shape is{x1_out.shape}')

    x3 = outLayer3_outconv
    x1 = x1_out

    x3 = rearrange((x3), 'n h w c -> n c h w')
    x1 = rearrange((x1), 'n h w c -> n c h w')
    return [x3,x1]


# basic building block
# no of kernels is the same as the number of output channels in tensorflow
# input_shape=(2,480,640,1) # (batch,height,width,number of channels)
# x=tf.random.normal(input_shape)
# feats_c,feats_f = Module1(x)
# feats_c = rearrange((feats_c), 'n h w c -> n c h w')
# feats_f = rearrange((feats_f), 'n h w c -> n c h w')
# print(f'Module 1 output is {feats_c.shape}')