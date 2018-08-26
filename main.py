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
logdir="./pkcp_logs/se/"

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

#%% model
tf.reset_default_graph()

inp = tf.placeholder(tf.float32, [None, size, size, 3], "input")
noisy = tf.constant(False)
noisy_img = inp
noisy_img = tf.map_fn(lambda x: tf.image.random_hue(x, 0.1), noisy_img)
noisy_img = tf.map_fn(lambda x: tf.image.random_brightness(x, 0.1), noisy_img)
noisy_img = tf.map_fn(lambda x: tf.image.random_contrast(x, 0.9, 1.1), noisy_img)
noisy_img = tf.map_fn(lambda x: tf.image.random_saturation(x, 0.7, 1.3), noisy_img)
noisy_img = noisy_img + tf.random_normal(tf.shape(inp), stddev=0.01)
img = tf.cond(noisy, lambda :noisy_img, lambda :inp)
label = tf.placeholder(tf.float32, [None, len(labels)], "label")
def residual(x, ch, activation_fn=tf.nn.relu):
    with tf.contrib.slim.arg_scope([tf.contrib.slim.conv2d], activation_fn=activation_fn), tf.name_scope("residual"):
        route1 = tf.contrib.slim.conv2d(x, ch // 4, 1)
        route2 = tf.contrib.slim.conv2d(x, ch // 4, 1)
        route2 = tf.contrib.slim.conv2d(route2, ch // 4, 3)
        route3 = tf.contrib.slim.conv2d(x, ch // 4, 1)
        route3 = tf.contrib.slim.conv2d(route3, ch // 4 + ch // 8, 3)
        route3 = tf.contrib.slim.conv2d(route3, ch // 2, 3)
        inception = tf.concat([route1, route2, route3], axis=-1)
        residual = tf.contrib.slim.conv2d(inception, ch, 1, activation_fn=None, normalizer_fn=None)

        with tf.name_scope("se"):
            se = tf.contrib.slim.avg_pool2d(residual, residual.shape[1:3])
            se = tf.contrib.slim.fully_connected(se, ch // 16)
            se = tf.contrib.slim.fully_connected(se, ch, activation_fn=tf.nn.sigmoid)
            se = tf.reshape(se, (-1, 1, 1, ch))
        x = residual * se + x
        x = activation_fn(x)
    return x


with tf.contrib.slim.arg_scope([tf.contrib.slim.conv2d, tf.contrib.slim.conv2d_transpose],
    activation_fn=functools.partial(tf.nn.leaky_relu, alpha=0.01),
    normalizer_fn=functools.partial(tf.contrib.slim.batch_norm, renorm=True)
    ):
    x = tf.contrib.slim.conv2d(img, 64, 3)
    w1 = x.graph.get_tensor_by_name(x.op.name.split("/")[-2] + "/weights:0")

    for i in range(5):
        x = residual(x, 64)

    x_ = tf.contrib.slim.max_pool2d(x, 2)
    x_ = tf.contrib.slim.conv2d(x_, 128, 1)
    x = x_ + tf.contrib.slim.conv2d(x, 128, 3, stride=2)

    for i in range(5):
        x = residual(x, 128)

    x_ = tf.contrib.slim.conv2d(x, 256, 4, padding="valid")
    x_ = tf.contrib.slim.max_pool2d(x_, 2)
    x_ = tf.contrib.slim.conv2d(x_, 256, 1)
    x = x_ + tf.contrib.slim.conv2d(x, 256, 5, stride=2, padding="valid")

    for i in range(5):
        x = residual(x, 256)

    x2 = tf.contrib.slim.avg_pool2d(x, x.shape[1:3])
    fc = tf.contrib.slim.flatten(x2)
    pred2 = tf.contrib.slim.fully_connected(fc, len(labels), activation_fn=tf.nn.softmax)

    x_ = tf.contrib.slim.max_pool2d(x, 3, stride=2)
    x_ = tf.contrib.slim.conv2d(x_, 512, 1)
    x = x_ + tf.contrib.slim.conv2d(x, 512, 3, stride=2, padding="valid")

    for i in range(5):
        x = residual(x, 512)

    x = tf.contrib.slim.conv2d(x, 1024, 3)
    x = tf.contrib.slim.avg_pool2d(x, x.shape[1:3])
    fc = tf.contrib.slim.flatten(x)
    pred = tf.contrib.slim.fully_connected(fc, len(labels), activation_fn=tf.nn.softmax)

with tf.name_scope("optimize"):
    loss = -tf.reduce_sum(label * tf.log(pred + 1e-10), axis=1)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # 今回は使ってない
    loss2 = -tf.reduce_sum(label * tf.log(pred2 + 1e-10), axis=1)
    loss2 = tf.reduce_mean(loss2)
    optimizer2 = tf.train.AdamOptimizer().minimize(loss2)

with tf.name_scope("summary"):
    log_img = tf.placeholder(tf.uint8, [len(labels), 32, 32, 3])
    loss_log = tf.summary.scalar("loss", loss)
    decoder_log = tf.summary.image("img", tf.cast(tf.map_fn(lambda x: tf.cast(log_img[x], tf.int64), tf.argmax(pred, 1)), tf.uint8), 10)
    result_log = tf.summary.merge([loss_log, decoder_log])
    input_log = tf.summary.image("img", img, 10)

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
saver = tf.train.Saver()
# saver.save(sess, "./pkcp_good/model")
saver.restore(sess, "./pkcp_good/model")

#%%
# 画像指定img_pathにパーティの画像
def debug_img(img=None):
    if img is None:
        img_path = "./sample.jpg"
        img = cv2.imread(img_path)
    return np.array([[(cv2.resize(poke, (size, size))) for poke in x] for x in get_pokemon_imgs(img)]) / 255.


# 候補画像のグループから、一番良いものを選択する
def clustering(tests):
    a = [sess.run(pred, feed_dict={inp: t}) for t in tests]
    index = np.max(a, axis=2).mean(axis=1).argmax()
    return a[index].argmax(axis=1), index

#%%
# 選択画面の画像
t_data = debug_img() # ./sample.jpg
# t_data = debug_img(img) # numpyデータを直接入れる際
numbers, index = clustering(t_data)


pylab.subplot(2, 1, 1)
pylab.axis("off")
pylab.imshow(pv(imgs[numbers,], 6, 1).astype(np.uint8))
pylab.subplot(2, 1, 2)
pylab.axis("off")
pylab.imshow(pv(t_data[index] * 255, 6, 1).astype(np.uint8))
pylab.show()
