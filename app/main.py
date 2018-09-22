import cv2
import numpy as np
import math
from scipy import ndimage
import tensorflow as tf
import pylab
from PIL import Image
import random
import itertools
import functools
import warnings
pi180 = np.pi / 180 * 360

#%%
# 輪郭取得
red, blue = min, max
def get_contour(img, team):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gaus = cv2.GaussianBlur(gray, (11, 11), 0)
    th2 = cv2.adaptiveThreshold(gaus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    contours = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = [x for x in contours if cv2.contourArea(x) > 60000]
    party = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        trim = img[y:y+h, x:x+w]
        party.append([np.sum(trim[:, :, 0] > trim[:, :, 1]), c])
    return team(party, key=lambda x:x[0])[1]


# 切り取られたパーティ画像をいい感じに回転させる処理
def adjast_img(img, team, rev=False):
    rect = cv2.minAreaRect(get_contour(img, team))
    box = cv2.boxPoints(rect)
    box = sorted(box, key=lambda x: x[1])

    target = [box[0], box[1]] if np.linalg.norm((box[0] - box[1]).astype(np.float64)) < np.linalg.norm((box[1] - box[2]).astype(np.float64)) else [box[0], box[2]]
    k = math.atan2(target[1][0] - target[0][0], target[1][1] - target[0][1])
    result = ndimage.rotate(img, 90 + math.degrees(k) * (1 if rev else -1))
    return result


# パーティ画像をまっすぐに回転させて切り取り
def trimming(img, team, rev=False):
    img = adjast_img(img, team, rev=rev)
    x, y, w, h = cv2.boundingRect(get_contour(img, team))
    return img[y:y+h, x:x+w]


# アイコンの切り取り
def crop(trimed):
    h, w = trimed.shape[:2]
    bh = h//7
    th = lambda x: bh + (h - bh) // 3 * x
    nar = 10
    img_size = (500, 500)

    pokemons = []
    for t in range(3):
        img1 = cv2.resize(trimed[th(t) + nar:th(t+1) - int(nar * 1.5), nar:w//2 - int(nar * 1.5)], img_size)
        img2 = cv2.resize(trimed[th(t) + nar:th(t+1) - int(nar * 1.5), nar + w//2:w - int(nar * 1.5)], img_size)
        pokemons.append(img1)
        pokemons.append(img2)
    return np.array(pokemons)[:, 10: 370, 80:440, ::-1]


def get_pokemon_imgs(party_img):
    orgHeight, orgWidth = party_img.shape[:2]

    area = orgHeight * orgWidth

    magni = 1000 / orgWidth
    img = cv2.resize(party_img, (int(orgWidth * magni), int(orgHeight * magni)))

    # パーティ全体の切り取り
    # trimedにパーティ全体が格納されている
    trimed1 = trimming(img, red)
    trimed2 = trimming(img, red, rev=True)

    return crop(trimed1), crop(trimed2)


#%% data
size = 50
def mapfunc(f):
    return lambda *data: np.array([f(*x) for x in zip(*data)])


def to_black(img):
    return img[:, :, :3] * (img[:, :, 3:] / 255)


def affine(img, p1, p2):
    p1 = np.asarray(p1).astype(np.float32)
    p2 = np.asarray(p2).astype(np.float32)
    w, h = img.shape[:2]
    M = cv2.getAffineTransform(p1, p2)
    tfimg = cv2.warpAffine(img, M, (w, h))
    return tfimg

def imread(number):
    img = Image.open("pokemon/{:03d}.png".format(number))
    img = np.array(img)
    img = to_black(img).astype(np.uint8)
    return img


def plt(imgs, hs, ws):
    h, w = imgs.shape[1 : 3]
    # view = np.zeros((h * hs, w * ws, 3))
    view = np.zeros((h * hs, w * ws))
    cnt = 0
    for i in range(hs):
        for j in range(ws):
            view[i * h : i * h + h, j * w : j * w + w] = imgs[cnt]
            cnt += 1
    return np.clip(view, 0, 1)


# labels = [
#         233, 445, 797, 785, 130, 681,
#         232, 444, 796, 784, 129, 680,
#         231, 443, 795, 783, 128, 679,
#         230, 442, 794, 782, 127, 678,
#         229, 441, 793, 781, 126, 677,
#         228, 440, 792, 780, 125, 676,
#         227, 439, 791, 779, 124, 675,
#         226, 438, 790, 778, 123, 674,
#         225, 437, 789, 777, 122, 673,
#         224, 436, 788, 776, 121, 672,
#         223, 435, 787, 775, 120, 671,
#         222, 434, 786, 774, 119, 670,
#         221, 433,   1, 773, 118, 669,
#         220, 432,   2, 772, 117, 668,
#         219, 431,   3, 771, 116, 667,
#         218, 430,   4, 770, 115, 666,
#         217, 429,   5, 769, 114, 665,
#         ]
labels = list(range(1, 807 + 1))

if len(labels) != len(set(labels)):
    warnings.warn("labels dupulicated!!!!!!!!")

def background(i, r):
    img = Image.open("backs/back{}.jpeg".format(i))
    img = img.convert("RGB")
    img = np.asarray(img)
    img = cv2.resize(img, (50, 50))

    center = (img.shape[0] // 2, img.shape[1] // 2)
    rotMat = cv2.getRotationMatrix2D(center, r, 1.0)
    rotated = cv2.warpAffine(img, rotMat, img.shape[0:2], flags=cv2.INTER_LINEAR)
    return rotated[25 - 32//2 : 25 + 32//2, 25 - 32//2 : 25 + 32//2]


def merge(back, img, mask):
    im0 = cv2.bitwise_and(img, mask)
    im1 = cv2.bitwise_and(back, cv2.bitwise_not(mask))
    return cv2.bitwise_or(im0, im1)


@mapfunc
def mask(gray_img):
    _, mask = cv2.threshold(np.asarray(gray_img), 0, 255, cv2.THRESH_BINARY)
    return cv2.merge((mask, mask, mask))


# %% create data
a = np.load("six_test70x70.npy")
b = cv2.resize(a[5], (32, 32))
item = np.array(b[19:30, 26:32])
h, w, _ = item.shape
item[:, w//2:] = item[:, :w//2 + (1 if w & 1 else 0)][:, ::-1]


@mapfunc
def noise(img):
    blur = [1, 3]
    h, w, _ = item.shape

    g = img

    hi = random.randint(0, 21)
    wi = random.randint(0, 26)
    g[hi:hi + h, wi:wi + w] = item
    g = cv2.GaussianBlur(g, (random.choice(blur), random.choice(blur)), 0)
    return g

targets = list(itertools.product(range(-1, 1 + 1), repeat=3))
random.shuffle(targets)
@mapfunc
def trans_noise(img):
    p = [[16, 0], [0, 32], [32, 32]]
    top, left, right = np.random.normal(loc=0., scale=1., size=3)
    img = affine(img, p, [[16, top], [left, 32], [32 + right, 32]])

    x, y = np.random.normal(loc=0., scale=2., size=2)
    return affine(img, p, [[16 + x, y], [x, 32 + y], [32 + x, 32 + y]])

# onehot = np.eye(len(labels), dtype=np.float32)
#
# img_indices = list(range(len(imgs)))

def proc(imgs, indices):
    r = random.getrandbits(30)
    noise_img = trans_noise(imgs[indices,])
    noise_gray = mapfunc(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY))(noise_img)
    noise_gray = noise_gray * (noise_gray > 25)

    back = background(np.random.randint(0, 6), np.random.randint(0, 360))
    fake = np.array([merge(back, m, g) for g, m in zip(mask(noise_gray), noise_img)])
    return list((mapfunc(lambda x: cv2.resize(x, (size, size)))(noise(np.array(fake))) / 255.).astype(np.float32))

#%% model
tf.reset_default_graph()

with tf.name_scope("input"):
    inp = tf.placeholder(tf.float32, [None, size, size, 3], "input")
    noisy = tf.constant(False)
    noisy_img = inp
    noisy_img = tf.map_fn(lambda x: tf.image.random_hue(x, 0.1), noisy_img)
    noisy_img = tf.map_fn(lambda x: tf.image.random_brightness(x, 0.1), noisy_img)
    noisy_img = tf.map_fn(lambda x: tf.image.random_contrast(x, 0.9, 1.1), noisy_img)
    noisy_img = tf.map_fn(lambda x: tf.image.random_saturation(x, 0.7, 1.3), noisy_img)
    noisy_img = noisy_img + tf.random_normal(tf.shape(inp), stddev=0.01)
    img = tf.cond(noisy, lambda :noisy_img, lambda :inp)
    rgb = img

    minv = tf.reduce_min(img, axis=[-1, -2, -3])
    for i in range(3):
        minv = tf.expand_dims(minv, 1)
    img -= minv
    maxv = tf.reduce_max(img, axis=[-1, -2, -3])
    for i in range(3):
        maxv = tf.expand_dims(maxv, 1)
    img /= maxv + 1e-10

    hsv = tf.image.rgb_to_hsv(img)
    t = hsv[..., 0] * pi180
    hs = tf.stack([tf.sin(t), tf.cos(t)], axis=-1)
    hs *= tf.expand_dims(hsv[..., 1], -1)

    label = tf.placeholder(tf.float32, [None, len(labels)], "label")

residual_cnt = 0
def residual(x, ch, activation_fn=tf.nn.relu, use_se=False):
    global residual_cnt
    with tf.variable_scope("conv_{}".format(residual_cnt)):
        with tf.contrib.slim.arg_scope([tf.contrib.slim.separable_conv2d], activation_fn=activation_fn), tf.variable_scope("xception_{}".format(residual_cnt)):
            conv = x
            conv = tf.contrib.slim.separable_conv2d(conv, ch, 3, depth_multiplier=1)
            conv = tf.contrib.slim.separable_conv2d(conv, ch, 3, depth_multiplier=1)
            conv = tf.contrib.slim.separable_conv2d(conv, ch, 3, depth_multiplier=1, activation_fn=None)

        with tf.variable_scope("se_{}".format(residual_cnt)):
            se = tf.keras.layers.GlobalAveragePooling2D()(conv)
            se = tf.contrib.slim.fully_connected(se, ch // 16)
            se = tf.contrib.slim.fully_connected(se, ch, activation_fn=tf.nn.sigmoid)
            se = tf.reshape(se, (-1, 1, 1, ch))

        with tf.name_scope("connect_{}".format(residual_cnt)):
            if use_se:
                x = conv * se + x
            else:
                x = conv * 0.2 + x
            x = activation_fn(x)
    residual_cnt += 1
    return x

rdep_cnt = 0
def rdep(layer, kernel_size, depth_multiplier, scale=0.2, dropout=None, activation_fn=tf.nn.relu, loop=0, bn=True):
    global rdep_cnt
    with tf.variable_scope("rdep_{}".format(rdep_cnt)):
        dep = tf.keras.layers.DepthwiseConv2D(kernel_size, depth_multiplier=depth_multiplier, padding="same", use_bias=not bn)(layer)
        for i in range(loop):
            with tf.variable_scope("loop_{}".format(i)):
                if bn:
                    dep = tf.contrib.slim.batch_norm(dep, renorm=True)
                dep = activation_fn(dep)
                dep = tf.keras.layers.DepthwiseConv2D(kernel_size, depth_multiplier=1, padding="same", use_bias=not bn)(dep)

        # dep = tf.keras.layers.DepthwiseConv2D([1, kernel_size], depth_multiplier=depth_multiplier, padding="same", use_bias=False)(layer)
        # dep = tf.contrib.slim.batch_norm(dep, renorm=True)
        # dep = activation_fn(dep)
        # dep = tf.keras.layers.DepthwiseConv2D([kernel_size, 1], depth_multiplier=1, padding="same", use_bias=False)(dep)
        if bn:
            dep = tf.contrib.slim.batch_norm(dep, renorm=True)

        if depth_multiplier == 1:
            dep = dep * scale + layer
        else:
            dep = tf.reshape(dep, (-1, *layer.shape[1:], depth_multiplier))
            dep = tf.transpose(dep, (0, 4, 1, 2, 3)) * scale + tf.expand_dims(layer, 1)
            dep = tf.transpose(dep, (0, 2, 3, 1, 4))
            dep = tf.reshape(dep, (-1, *layer.shape[1:-1], layer.shape[-1] * depth_multiplier))

        if dropout is not None:
            dep = tf.cond(noisy, lambda: dropout(dep), lambda: dep)
        dep = activation_fn(dep)
    rdep_cnt += 1
    return dep

rpool_cnt = 0
def rpool(layer, kernel_size, depth_multiplier, stride=2, scale=0.2, padding="same", dropout=None, activation_fn=tf.nn.relu, loop=0, bn=True):
    global rpool_cnt
    with tf.variable_scope("rpool_{}".format(rpool_cnt)):
        pool = tf.contrib.slim.max_pool2d(layer, kernel_size, stride=stride, padding=padding)
        first, second = (depth_multiplier + 1) // 2, depth_multiplier // 2

        dep = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=first, padding=padding, use_bias=not bn)(layer)
        for i in range(loop):
            with tf.variable_scope("loop_1_{}".format(i)):
                if bn:
                    dep = tf.contrib.slim.batch_norm(dep, renorm=True)
                dep = activation_fn(dep)
                dep = tf.keras.layers.DepthwiseConv2D(kernel_size, depth_multiplier=1, padding="same", use_bias=not bn)(dep)
        dep = tf.reshape(dep, (-1, *pool.shape[1:], first))
        dep1 = dep

        dep = tf.keras.layers.DepthwiseConv2D(kernel_size, depth_multiplier=second, padding="same", use_bias=not bn)(pool)
        for i in range(loop):
            with tf.variable_scope("loop_2_{}".format(i)):
                if bn:
                    dep = tf.contrib.slim.batch_norm(dep, renorm=True)
                dep = activation_fn(dep)
                dep = tf.keras.layers.DepthwiseConv2D(kernel_size, depth_multiplier=1, padding="same", use_bias=not bn)(dep)
        dep = tf.reshape(dep, (-1, *pool.shape[1:], second))
        dep2 = dep

        dep = tf.concat([dep1, dep2], axis=-1)

        if bn:
            dep = tf.contrib.slim.batch_norm(dep, renorm=True)

        dep = tf.transpose(dep, (0, 4, 1, 2, 3)) * scale + tf.expand_dims(pool, 1)
        dep = tf.transpose(dep, (0, 2, 3, 1, 4))
        dep = tf.reshape(dep, (-1, *pool.shape[1:-1], pool.shape[-1] * depth_multiplier))
        if dropout is not None:
            dep = tf.cond(noisy, lambda: dropout(dep), lambda: dep)
        dep = activation_fn(dep)
    rpool_cnt += 1
    return dep

pw_cnt = 0
def pointwise(layer, out_ch, **kwargs):
    global pw_cnt
    with tf.variable_scope("pointwise_{}".format(pw_cnt)):
        layer = tf.contrib.slim.conv2d(layer, out_ch, 1, padding="same", **kwargs)
    pw_cnt += 1
    return layer

with tf.contrib.slim.arg_scope([tf.contrib.slim.separable_conv2d, tf.contrib.slim.conv2d], normalizer_fn=functools.partial(tf.contrib.slim.batch_norm, renorm=True)):
    with tf.variable_scope("rgb"):
        act = tf.nn.relu
        x = rgb
        b1 = rdep(x, 3, 15, scale=1., activation_fn=act)
        b2 = rdep(x, 5, 15, scale=1., activation_fn=act)
        x = tf.concat([b1, b2], axis=-1)
        x1 = x
        x.shape
        x = rpool(x, 3, 4, activation_fn=act)
        x2 = x
        x.shape
        x = rpool(x, 3, 2, padding="valid", activation_fn=act)
        x3 = x
        x.shape
        x = pointwise(x, 256, activation_fn=act)
        x4 = x
        x.shape
        route1 = x

        with tf.variable_scope("block"):
            for i in range(6):
                x = residual(x, 256, use_se=True, activation_fn=act)

        x = rpool(x, 3, 4, activation_fn=act)
        route1_2 = x
        x.shape

        with tf.variable_scope("pred"):
            x = tf.contrib.slim.avg_pool2d(x, x.shape[1:3])
            fc = tf.contrib.slim.flatten(x)
            pred_rgb = tf.contrib.slim.fully_connected(fc, len(labels), activation_fn=tf.nn.softmax)

    with tf.variable_scope("hs"):
        act = tf.nn.relu
        x = hs
        x = rdep(x, 3, 8, scale=1., activation_fn=act)
        # x1 = x
        # x.shape
        # x = rpool(x, 3, 16, activation_fn=act, loop=2)
        x = rpool(x, 3, 8, activation_fn=act)
        x2 = x
        x.shape
        x = rpool(x, 3, 6, padding="valid", activation_fn=act)
        x3 = x
        x.shape
        x = pointwise(x, 256, activation_fn=act)
        x4 = x
        x.shape
        route2 = x

        with tf.variable_scope("block"):
            for i in range(6):
                x = residual(x, 256, use_se=True, activation_fn=act)

        x = rpool(x, 3, 4, activation_fn=act)
        route2_2 = x
        x.shape

        with tf.variable_scope("pred"):
            x = tf.contrib.slim.avg_pool2d(x, x.shape[1:3])
            fc = tf.contrib.slim.flatten(x)
            pred_hs = tf.contrib.slim.fully_connected(fc, len(labels), activation_fn=tf.nn.softmax)

    with tf.variable_scope("mix"):
        act = tf.nn.relu
        x = route1 + route2
        with tf.variable_scope("block"):
            for i in range(2):
                x = residual(x, 256, use_se=True, activation_fn=act)

        x = rpool(x, 3, 4, activation_fn=act)

        x = tf.concat([route1_2, route2_2, x], axis=-1)
        x = pointwise(x, 1024)
        x.shape

        with tf.variable_scope("pred"):
            x = tf.contrib.slim.avg_pool2d(x, x.shape[1:3])
            fc = tf.contrib.slim.flatten(x)
            pred_mix = tf.contrib.slim.fully_connected(fc, len(labels), activation_fn=tf.nn.softmax)

# rgb_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rgb")
# hs_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hs")
# mix_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mix")
# with tf.name_scope("optimize"):
#     loss = -tf.reduce_sum(label * tf.log(pred_rgb + 1e-10), axis=1)
#     loss_rgb = tf.reduce_mean(loss)
#     optimizer_rgb = tf.train.AdamOptimizer().minimize(loss_rgb, var_list=rgb_vars)
#
#     loss = -tf.reduce_sum(label * tf.log(pred_hs + 1e-10), axis=1)
#     loss_hs = tf.reduce_mean(loss)
#     optimizer_hs = tf.train.AdamOptimizer().minimize(loss_hs, var_list=hs_vars)
#
#     loss = -tf.reduce_sum(label * tf.log(pred_mix + 1e-10), axis=1)
#     loss_mix = tf.reduce_mean(loss)
#     optimizer_mix = tf.train.AdamOptimizer().minimize(loss_mix, var_list=mix_vars)

#%%
step = 0
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "./pkcp_good/model")

#%% view labels
def pv(imgs, w, h, ch=3):
    sh, sw = imgs.shape[1:3]
    draw = np.zeros((h * sh, w * sw, ch))
    try:
        draw = np.squeeze(draw, -1)
    except: pass
    c = 0
    for i in range(h):
        for j in range(w):
            draw[i * sh : (i + 1) * sh, j * sw : (j + 1) * sw] = imgs[c]
            c += 1
    return draw

#%%
# 画像指定img_pathにパーティの画像
def debug_img(img=None):
    if img is None:
        img_path = "./sample.jpg"
        img = cv2.imread(img_path)
    return np.array([[(cv2.resize(poke, (size, size))) for poke in x] for x in get_pokemon_imgs(img)]) / 255.


# 候補画像のグループから、一番良いものを選択する
def clustering(tests):
    a = [sess.run(pred_mix, feed_dict={inp: t}) for t in tests]
    index = np.max(a, axis=2).mean(axis=1).argmax()
    return a[index].argmax(axis=1), index

imgs = np.array([cv2.resize(imread(x), (32, 32)) for x in labels])
#%%
# 選択画面の画像
t_data = debug_img() # ./sample.jpg
# t_data = debug_img(img) # numpyデータを直接入れる際
numbers, index = clustering(t_data)

# print(numbers + 1)

pylab.subplot(2, 1, 1)
pylab.axis("off")
pylab.imshow(pv(imgs[numbers,], 6, 1).astype(np.uint8))
pylab.subplot(2, 1, 2)
pylab.axis("off")
pylab.imshow(pv(t_data[index] * 255, 6, 1).astype(np.uint8))
pylab.show()
