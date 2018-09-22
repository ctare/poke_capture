import tensorflow as tf
import numpy as np
import pylab
from PIL import Image
import cv2
import random
import itertools
import functools
from tqdm import tqdm
import warnings
from glob import glob

#%% data
pi180 = np.pi / 180 * 360
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

#create test data
test_data = np.array([(cv2.resize(x, (size, size))) for x in np.load("six_test70x70.npy")]) / 255.
test_label = np.array([796, 129, 680, 444, 784, 232])

ext_test_data = np.ones((0, 50, 50, 3))
ext_test_label = np.ones((0, ))
for filename in glob("pkcp_test_data/*"):
    (_, l), (_, d) = np.load(filename).items()
    ext_test_data = np.concatenate((ext_test_data, d))
    ext_test_label = np.concatenate((ext_test_label, l))
test_data = np.concatenate([test_data, ext_test_data])
test_label = np.concatenate([test_label, ext_test_label]).astype(int)
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
#
#         # 184, 444, 443, 552, 144, 117,
#         #mrl gbyt hkml wrbr frzr shdr
#         ] + list(range(6, 114))
labels = list(range(1, 807 + 1))

# labels = np.array(list(set(np.concatenate((test_label, test_label - 1)))))
# test_label = (labels == np.array([test_label]).T).argmax(axis=1)

# pylab.imshow(test_data[0])
# pylab.imshow(imgs[labels[test_label[0]]])

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
# target = 10
# target_img = np.array(imgs[target : target + 6])
# target_img = test_data
# dec_test_imgs, lossv = sess.run([decoder, loss], feed_dict={inp: target_img})
# t, c = 20, 0
# # pylab.imshow(plt(dec_test_imgs[..., t : t + [1, 3][c]], 2, 3))
# pylab.title("loss: {}".format(lossv))
# pylab.imshow(plt(np.squeeze(dec_test_imgs), 2, 3))
# pylab.show()
# pylab.imshow(plt(np.squeeze(target_img), 2, 3))
# pylab.show()

# %% create data

a = np.load("six_test70x70.npy")
b = cv2.resize(a[5], (32, 32))
item = np.array(b[19:30, 26:32])
h, w, _ = item.shape
item[:, w//2:] = item[:, :w//2 + (1 if w & 1 else 0)][:, ::-1]

# def lgblur(img, times=10):
#     img_shape = img.shape
#     img = cv2.resize(img, (size, size))
#     img = np.expand_dims(img.transpose(2, 0, 1), -1)
#     lg = (np.random.uniform(0, 1, (1, size, size, 1)) < 0.2).astype(np.int8)
#     for i in (range(times)):
#         lg, img = sess.run([next_generation, mix_image], feed_dict={lifegame: lg, mix_target: img})
#     img = np.squeeze(img).transpose(1, 2, 0)
#     img = cv2.resize(img, img_shape[:2])
#     return np.squeeze(img)

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

imgs = np.array([cv2.resize(imread(x), (32, 32)) for x in labels])
onehot = np.eye(len(labels), dtype=np.float32)

img_indices = list(range(len(imgs)))

def proc(imgs, indices):
    r = random.getrandbits(30)
    noise_img = trans_noise(imgs[indices,])
    noise_gray = mapfunc(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY))(noise_img)
    noise_gray = noise_gray * (noise_gray > 25)

    back = background(np.random.randint(0, 6), np.random.randint(0, 360))
    fake = np.array([merge(back, m, g) for g, m in zip(mask(noise_gray), noise_img)])
    return list((mapfunc(lambda x: cv2.resize(x, (size, size)))(noise(np.array(fake))) / 255.).astype(np.float32))

# #%%
# h1, h2, im = sess.run([hsv1, hsv2, img], feed_dict={inp: proc(imgs, [58,]), noisy: True}); h1, h2, im = h1[0], h2[0], im[0]
# h2.min(), h2.max(), h2.mean()
# pylab.imshow(np.squeeze(h2))
# pylab.imshow(np.concatenate([h1, h2], axis=-1))

#%% model
tf.reset_default_graph()

# with tf.name_scope("lifegame"):
#     sz = 100
#
#     mix_target = tf.placeholder(tf.float32, [None, sz, sz, 1], name="mix_target")
#     lifegame = tf.placeholder(tf.int8, [1, sz, sz, 1], name="lifegame")
#
#     with tf.name_scope("count"):
#         flt = tf.initializers.constant(np.array([
#         [1, 1, 1],
#         [1, 0, 1],
#         [1, 1, 1],
#         ]))
#         float_lifegame = tf.cast(lifegame, tf.float32)
#         cnt = tf.contrib.slim.conv2d(float_lifegame, 1, 3, padding="same", activation_fn=None, weights_initializer=flt)
#         input_cnt = tf.contrib.slim.conv2d(mix_target * float_lifegame, 1, 3, padding="same", activation_fn=None, weights_initializer=flt)
#
#     with tf.name_scope("next_generation"):
#         keep = tf.equal(lifegame, 1) & tf.greater_equal(cnt, 2) & tf.less_equal(cnt, 3)
#         born = tf.equal(lifegame, 0) & tf.equal(cnt, 3)
#         next_generation = keep | born
#
#     with tf.name_scope("image_mix"):
#         target = keep # keep, born, next_generation
#         mix_image = tf.cast(target, tf.float32) * input_cnt / tf.maximum(cnt, 1)
#         mix_image = tf.cast(tf.logical_not(target), tf.float32) * mix_target + mix_image

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

    # hsv = tf.concat([img, tf.image.rgb_to_hsv(img)], axis=-1)
    hsv = tf.image.rgb_to_hsv(img)
    t = hsv[..., 0] * pi180
    hs = tf.stack([tf.sin(t), tf.cos(t)], axis=-1)
    hs *= tf.expand_dims(hsv[..., 1], -1)
    # hsv = tf.concat([hsv_yx, hsv[..., 1:]], axis=-1)

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

selection_cnt = 0
def selection(*activations):
    def activation(x):
        with tf.variable_scope("selection_{}".format(selection_cnt)):
            route = [act(x) for act in activations]
            x = tf.stack(route, axis=-1)
            alpha = tf.Variable(tf.ones((x.shape[-2], x.shape[-1],)))
            alpha = tf.nn.softmax(alpha, axis=-1)
            x = tf.reduce_sum(x * alpha, axis=-1)
        return x
    return activation


with tf.contrib.slim.arg_scope([tf.contrib.slim.separable_conv2d, tf.contrib.slim.conv2d],
    # activation_fn=functools.partial(tf.nn.leaky_relu, alpha=0.01),
    # activation_fn=lambda x: tf.keras.layers.PReLU()(x),
    normalizer_fn=functools.partial(tf.contrib.slim.batch_norm, renorm=True)), tf.contrib.slim.arg_scope([tf.contrib.slim.conv2d],
    # activation_fn=lambda x: tf.keras.layers.PReLU()(x),
    # activation_fn=tf.nn.leaky_relu,
    ):
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

        # x = pointwise(x, 512)
        x = rpool(x, 3, 4, activation_fn=act)
        route1_2 = x
        x.shape

        with tf.variable_scope("pred"):
            # x = rdep(x, 3, 2)
            # x = tf.contrib.slim.conv2d(x, 1024, 1)
            x = tf.contrib.slim.avg_pool2d(x, x.shape[1:3])
            fc = tf.contrib.slim.flatten(x)
            pred_rgb = tf.contrib.slim.fully_connected(fc, len(labels), activation_fn=tf.nn.softmax)

    with tf.variable_scope("hs"):
        # act = selection(tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid)
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

        # x = pointwise(x, 512)
        x = rpool(x, 3, 4, activation_fn=act)
        route2_2 = x
        x.shape

        with tf.variable_scope("pred"):
            # x = rdep(x, 3, 2)
            # x = tf.contrib.slim.conv2d(x, 1024, 1)
            x = tf.contrib.slim.avg_pool2d(x, x.shape[1:3])
            fc = tf.contrib.slim.flatten(x)
            pred_hs = tf.contrib.slim.fully_connected(fc, len(labels), activation_fn=tf.nn.softmax)

    with tf.variable_scope("mix"):
        act = tf.nn.relu
        x = route1 + route2 # tf.concat([route1, route2], axis=-1) and pointwise
        with tf.variable_scope("block"):
            for i in range(2):
                x = residual(x, 256, use_se=True, activation_fn=act)


        # x = pointwise(x, 512)
        x = rpool(x, 3, 4, activation_fn=act)

        x = tf.concat([route1_2, route2_2, x], axis=-1)
        x = pointwise(x, 1024)
        x.shape

        with tf.variable_scope("pred"):
            # x = rdep(x, 3, 2)
            # x = tf.contrib.slim.conv2d(x, 1024, 1)
            x = tf.contrib.slim.avg_pool2d(x, x.shape[1:3])
            fc = tf.contrib.slim.flatten(x)
            pred_mix = tf.contrib.slim.fully_connected(fc, len(labels), activation_fn=tf.nn.softmax)

rgb_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rgb")
hs_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hs")
mix_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mix")
with tf.name_scope("optimize"):
    loss = -tf.reduce_sum(label * tf.log(pred_rgb + 1e-10), axis=1)
    loss_rgb = tf.reduce_mean(loss)
    optimizer_rgb = tf.train.AdamOptimizer().minimize(loss_rgb, var_list=rgb_vars)

    loss = -tf.reduce_sum(label * tf.log(pred_hs + 1e-10), axis=1)
    loss_hs = tf.reduce_mean(loss)
    optimizer_hs = tf.train.AdamOptimizer().minimize(loss_hs, var_list=hs_vars)

    loss = -tf.reduce_sum(label * tf.log(pred_mix + 1e-10), axis=1)
    loss_mix = tf.reduce_mean(loss)
    optimizer_mix = tf.train.AdamOptimizer().minimize(loss_mix, var_list=mix_vars)


def add_log(pred, loss, name):
    acc = tf.equal(tf.argmax(pred, axis=1), tf.argmax(label, axis=1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    return tf.summary.scalar("acc_{}".format(name), acc), tf.summary.scalar("loss_{}".format(name), loss), acc

with tf.name_scope("summary"):
    acc_log_rgb, loss_log_rgb, acc_rgb = add_log(pred_rgb, loss_rgb, "rgb")
    acc_log_hs, loss_log_hs, acc_hs = add_log(pred_hs, loss_hs, "hs")
    acc_log_mix, loss_log_mix, acc_mix = add_log(pred_mix, loss_mix, "mix")

    log_img = tf.placeholder(tf.uint8, [len(labels), 32, 32, 3])
    decoder_log = tf.summary.image("img", tf.cast(tf.map_fn(lambda x: tf.cast(log_img[x], tf.int64), tf.argmax(pred_mix, 1)), tf.uint8), 10)
    # decoder_log = tf.summary.image("img", tf.cast(tf.map_fn(lambda x: tf.cast(log_img[x], tf.int64), tf.argmax(pred_rgb, 1)), tf.uint8), 10)
    result_log = tf.summary.merge_all()
    input_log = tf.summary.image("img", img, 10)

# logdir="./pkcp_logs_small/t105_rdep_rpool_se_sep_inception_minmax/"
# logdir="./pkcp_logs_small_3way/t105_goodmix_all_hsv_mini_mix/"
logdir="./pkcp_logs_small_3way/t105_tmp/"

#%%
step = 0
sess = tf.Session()
sess.run(tf.global_variables_initializer())

try:
    tf.gfile.DeleteRecursively(logdir)
except:
    pass

main_summary = tf.summary.FileWriter(logdir, sess.graph)
train_summary = tf.summary.FileWriter(logdir + "train")
train_input_summary = tf.summary.FileWriter(logdir + "train-input")
test_summary = tf.summary.FileWriter(logdir + "test")
test_input_summary = tf.summary.FileWriter(logdir + "test-input")

summaries = [main_summary, train_summary, train_input_summary, test_summary, test_input_summary]

#%% train
# true_label_indices = [2, 4, 5, 1, 3, 0]
# true_label_indices = [232, 444, 796, 784, 129, 680]
# true_label_indices = [796, 129, 680, 444, 784, 232]
# t_ = proc(imgs, true_label_indices)

# im = sess.run(tf.imagergb_to_hsv(img), feed_dict={inp: [cv2.resize(imgs[39], (50, 50))], noisy:True})[0]
# # im[..., 0] = 0.11
# pylab.imshow(im[..., 0])
# im_ = sess.run(tf.image.hsv_to_rgb(img), feed_dict={img: [im]})[0]
# pylab.imshow(im_.astype(np.uint8)) and pylab.show()

true_label = onehot[test_label,]
batch_n = 128
# batch_n = len(labels)
with tf.device("/device:GPU:0"):
    for epoch in (range(2000)):
        random.shuffle(img_indices)

        for target in zip(*[iter(img_indices)]* batch_n):
            data = proc(imgs, target)

            # sess.run([optimizer_rgb, optimizer_hs, optimizer_mix], feed_dict={inp: data, label: onehot[target,], noisy: True})
            sess.run([optimizer_mix], feed_dict={inp: data, label: onehot[target,], noisy: True})
            # sess.run([optimizer_rgb], feed_dict={inp: data, label: onehot[target,], noisy: True})
            # sess.run([optimizer_hs], feed_dict={inp: data, label: onehot[target,], noisy: True})

        if epoch % 100 == 0:
            random.shuffle(img_indices)
            target = img_indices[:105]
            t = proc(imgs, target)
            result, result_input = sess.run([result_log, input_log], feed_dict={inp: t, label: onehot[target,], log_img: imgs})
            train_summary.add_summary(result, step)
            train_input_summary.add_summary(result_input, step)

        t = test_data
        result, result_input, accuracy = sess.run([result_log, input_log, acc_mix], feed_dict={inp: t, label: true_label, log_img: imgs})
        if step >= 1500 and accuracy == 1:
            print("stop", step)
            break

        test_summary.add_summary(result, step)
        test_input_summary.add_summary(result_input, step)

        if epoch % 100 == 0:
            for i in summaries:
                i.flush()
        step += 1
print("ok")

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
            try:
                draw[i * sh : (i + 1) * sh, j * sw : (j + 1) * sw] = imgs[c]
            except:
                break
            c += 1
    return draw

pylab.imshow(pv(imgs, 6, 3).astype(np.uint8))

#%%
t_data1 = np.array([(cv2.resize(x, (size, size))) for x in pdata1]) / 255.
t_data2 = np.array([(cv2.resize(x, (size, size))) for x in pdata2]) / 255.
t_data = [t_data1, t_data2]
# t_data = test_data

# a = positioned_data.argmax(axis=1)
a = [sess.run(pred, feed_dict={inp: t}) for t in t_data]
index = np.max(a, axis=2).mean(axis=1).argmax()
a = a[index].argmax(axis=1)

pylab.imshow(pv(imgs[a,], 6, 1).astype(np.uint8))
pylab.show()
pylab.imshow(pv(t_data[index] * 255, 6, 1).astype(np.uint8))
pylab.show()
# pylab.imshow(pv(imgs[773:780], 6, 1).astype(np.uint8))
# pylab.show()
# 774

#%% view hidden layer outputs
a = sess.run("rdep_3/selection_11/Sum:0", feed_dict={inp: test_data[:1]})[0]
a -= a.min()
a /= a.max()
pylab.imshow(a[..., 8])

# #%%
# saver = tf.train.Saver()
# # saver.save(sess, "./pkcp_rdep_se/model")
# saver.restore(sess, "./pkcp_rdep_se/model")


#%%
target = pred_mix
pred_label = sess.run(target, feed_dict={inp: t, log_img: imgs}).argmax(axis=1)
miss = np.where(pred_label != true_label.argmax(axis=1))
pylab.imshow(pv(t[miss], len(miss[0]), 1))
pylab.show()
pylab.imshow(pv(imgs[pred_label[miss]], len(miss[0]), 1).astype(np.uint8))
pylab.show()
print(miss) # (array([ 6, 52]),)
pl = sess.run(target, feed_dict={inp: t, log_img: imgs})
pl[miss].max(axis=-1)

#%% saver
rgb_saver = tf.train.Saver(var_list=rgb_vars)
hs_saver = tf.train.Saver(var_list=hs_vars)
mix_saver = tf.train.Saver(var_list=mix_vars)

#%% save restore
rgb_saver.restore(sess, "way3/base/rgb")
hs_saver.restore(sess, "way3/base/hs")
mix_saver.restore(sess, "way3/base/mix")

# rgb_saver.save(sess, "way3/base/rgb")
# hs_saver.save(sess, "way3/base/hs")
# mix_saver.save(sess, "way3/base/mix")

#%% save all
saver = tf.train.Saver()
saver.save(sess, "way3/base/all")
