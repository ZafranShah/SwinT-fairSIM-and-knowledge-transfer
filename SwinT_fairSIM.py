#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:05:23 2023

@author: shah
"""


from tensorflow.keras.layers import Dense, Layer
import tensorflow as tf
import numpy as np
import collections
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dropout,
    Softmax,
    LayerNormalization,
    Conv2D,
    Conv2DTranspose,
    Layer,
    Activation
)



def normalize(epsilon = 1e-5, **kwargs):
    return tf.keras.layers.LayerNormalization(epsilon = epsilon, **kwargs)

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

class TruncatedDense(Dense):
    def __init__(self, units, use_bias=True, initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)):
        super().__init__(units, use_bias=use_bias, kernel_initializer=initializer)


class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.keras.activations.gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = TruncatedDense(hidden_features)
        self.act = act_layer
        self.fc2 = TruncatedDense(out_features)
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

def window_partition(x, size):

    h, w, ch = tf.keras.backend.int_shape(x)[-3:]
    
    out = tf.reshape(x, [-1, h // size, size, w // size, size, ch])
    

    
    window = tf.reshape(tf.transpose(out, [0, 1, 3, 2, 4, 5]),[-1,h // size*w // size ,size, size, ch])

    return window

def window_reverse(window, h, w):
    size, ch = tf.keras.backend.int_shape(window)[-2:]
    out = tf.reshape(window, [-1, h // size, w // size, size, size, ch])
    out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(out, shape=[-1, h, w, ch])
    return x


class WindowAttention(tf.keras.layers.Layer):


    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.proj_drop =proj_drop
        self.attn_drop= attn_drop
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        
    def build(self, dim):
        initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)

        self.relative_position_bias_table = tf.Variable(
            initializer(shape=((2*self.window_size[0]-1) * (2*self.window_size[1]-1), self.num_heads)), name="relative_position_bias_table")  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = tf.reshape(coords, [2, -1])  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = tf.transpose(relative_coords, perm=[1,2,0]) # Wh*Ww, Wh*Ww, 2
        relative_coords = relative_coords + [self.window_size[0] - 1, self.window_size[1] - 1]  # shift to start from 0
        relative_coords = relative_coords * [2*self.window_size[1] - 1, 1]
        self.relative_position_index = tf.math.reduce_sum(relative_coords,-1)  # Wh*Ww, Wh*Ww

        self.qkv = TruncatedDense(dim[-1] * 3, use_bias=self.qkv_bias)
        self.attn_drop = Dropout(self.attn_drop)
        self.proj = TruncatedDense(dim[-1])
        self.proj_drop = Dropout(self.proj_drop)
        self.softmax = Softmax(axis=-1)
        
        self.built = True

    def call(self, x, mask=None):

        B_, N, C = x.shape
        qkv = tf.transpose(tf.reshape(self.qkv(x), [-1, N, 3, self.num_heads, C // self.num_heads]), perm=[2, 0, 3, 1, 4]) # [3, B_, num_head, Ww*Wh, C//num_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ tf.transpose(k, [0, 1, 3, 2]))
        
        relative_position_bias = tf.reshape(tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, [-1])),
            [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias

        if mask is not None:
            nW = tf.keras.backend.int_shape(mask)[0] # every window has different mask [nW, N, N]
            attn = tf.reshape(attn, [-1, nW, self.num_heads, N, N]) +  tf.cast(tf.expand_dims(tf.expand_dims(mask, axis = 1), axis = 0), attn.dtype)
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = tf.reshape(tf.transpose(attn @ v, perm=[0, 2, 1, 3]), [-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    
class DropPath(tf.keras.layers.Layer):
    def __init__(self, rate = 0., **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.rate = rate
        
    def drop_path(self, x, rate):
        keep_prob = 1 - rate
        shape = tf.shape(x)
        w = keep_prob + tf.random.uniform([shape[0]] +  [1] * (len(shape) - 1), dtype = x.dtype)
        return tf.math.divide(x, keep_prob) * tf.floor(w)

    def call(self, inputs):
        out = inputs
        if 0 < self.rate and self.trainable:
            out = self.drop_path(inputs, self.rate)
        return out
    
    def get_config(self):
        config = super(DropPath, self).get_config()
        config["rate"] = self.rate
        return config

class SwinTransformerBlock(tf.keras.layers.Layer):


    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=tf.keras.activations.gelu, norm_layer=normalize):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_layer = normalize
        self.act_layer =act_layer
        self.drop_path= drop_path
        self.drop =drop
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        

        self.norm1 = normalize(name = "norm1")

        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        if 0 < drop_path:
            self.droppath = DropPath(drop_path, name = "drop_path")        

        self.norm2 = normalize(name = "norm2")
        
        
        

          
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=self.act_layer, drop=self.drop)
        
        self.attn_mask=None

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

            
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
            
        H, W = x_size
        img_mask = np.zeros([1, H, W, 1])  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
                
        img_mask = tf.constant(img_mask)
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
        attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
        attn_mask = tf.where(attn_mask==0, -100., 0,attn_mask)
        attn_mask = tf.where(attn_mask == 0, 0, attn_mask)
        return attn_mask

    def call(self, x, x_size):

        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        

        x = tf.reshape(x, [-1, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = tf.reshape(x_windows, [-1, self.window_size * self.window_size, C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows =self.attn(x_windows, mask=self.calculate_mask(x_size))
        # merge windows
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=(1, 2))
        else:
            x = shifted_x
        out = tf.reshape(x, [-1, H * W, C])

        # FFN
 
        if 0 < self.drop_path:
            out = self.droppath(out)
        out = shortcut + out
        out = self.mlp(self.norm2(out)) 
        if 0 < self.drop_path:
            out = self.droppath(out)
        out = shortcut + out
        

        
        return out

class PatchMerging(tf.keras.layers.Layer):


    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = norm_layer(epsilon=1e-5)

    def build(self, dim):
        self.reduction = TruncatedDense(2 * dim, use_bias=False)

    def call(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
#        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, [B, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = tf.reshape(x, [B, -1, 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(tf.keras.layers.Layer):


    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks =[SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
                       for i in range(depth)]

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def call(self, x, x_size):
        for blk in self.blocks:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(tf.keras.layers.Layer):

    def __init__(self, img_size=224, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, **kawargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def call(self, x):
        B, H, W, C = x.shape
        out = tf.reshape(x, [-1 ,H*W, self.embed_dim])        
        return out

class PatchUnEmbed(tf.keras.layers.Layer):


    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def call(self, x, x_size):
        B, HW, C = x.shape
        x = tf.reshape(x, [-1, x_size[0], x_size[1], self.embed_dim])  # B Ph*Pw C 
        return x

    
class ResidualSwinTransformerBlock(tf.keras.layers.Layer):


    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4):
        super(ResidualSwinTransformerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)


        self.conv = Conv2D(dim, 3, strides=1, padding= 'same')



        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def call(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group.call(x, x_size), x_size))) + x


     
    
    
    
class SwinT_fairSIM(tf.keras.layers.Layer):


    def __init__(self, img_size=256, patch_size=1, in_chans=1,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=normalize, ape=False, patch_norm=True,
                 use_checkpoint=False,
                 **kwargs):
        super(SwinT_fairSIM, self).__init__()

        self.mean = tf.zeros([1,1])
        #####################################################################################################
        ################################### 1, Compression head block ###################################
        self.conv_first = Conv2D(embed_dim//4, 3, strides=1, padding= 'same')
        self.conv_second = Conv2D(embed_dim//2, 3, strides=2, padding= 'same')
        self.conv_third = Conv2D(embed_dim, 3, strides=2, padding= 'same')


        #####################################################################################################
        #####################################################################################################
        ################################### 2, Swin Transformer ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = tf.Variable(tf.zeros(1, num_patches, embed_dim))
            TruncatedDense(self.absolute_pos_embed, std=.02)

        self.pos_drop = Dropout(drop_rate)
        
                

        # stochastic depth
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # build Residual Swin Transformer blocks (RSTB)
        self.Layers = []
        for i_layer in range(self.num_layers):
            layer = ResidualSwinTransformerBlock(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         )
            self.Layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction   strides=1, padding= (1,1)
        self.conv_after_body = Conv2D(embed_dim, 3, strides=1, padding= 'same')
        #####################################################################################################
        #####################################################################################################
        ################################### 3, Decompression Tail block###################################
        self.deconv_first = Conv2DTranspose(embed_dim//4, 3, strides=2, padding= 'same')
        self.deconv_second = Conv2DTranspose(embed_dim//2, 3, strides=2, padding= 'same')
        self.deconv_third = Conv2DTranspose(embed_dim, 3, strides=1, padding= 'same')
        
        self.conv_last = Conv2D(1, 1, strides=1, padding= 'same')

    def forward_features(self, x):
        x_size = (x.shape[1], x.shape[2])
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.Layers:
 
            x = layer(x, x_size)
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def call(self,raw_shape):        
        img_input = tf.keras.layers.Input(raw_shape)
        x_first = self.conv_first(img_input)
        x_second = self.conv_second(x_first)
        x_third = self.conv_third(x_second)
        
        


        res = self.conv_after_body(self.forward_features(x_third)) + x_third
            
        y_third= self.deconv_third(res) + x_third
        y_second=self.deconv_second(y_third) + x_second
        y_final=self.deconv_first(y_second) + x_first
            

        con_final=self.conv_last(y_final)
        output_layer = tf.keras.layers.Conv2D(1, 1, padding='same', strides=1, activation="relu")(con_final)

        
        model = tf.keras.Model(img_input, output_layer)
        return model
    
    
    def build_graph(self, raw_shape):
         x = tf.keras.layers.Input(shape=raw_shape)
         model= Model(inputs=[x], outputs=self.call(x))
         return model.summary()




