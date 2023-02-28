import tensorflow as tf
import os
import numpy as np
from PIL import Image
import shutil
import cv2
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl

# 可以显示中文
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

# 垃圾类名
class_names = ['0', '1', '2', '3', '4']


# 数据加载，按照8:2的比例加载垃圾数据
def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


# 测试mobilenet的准确率
def test_mobilenet():
    train_ds, val_ds, class_names = data_load("C:\Project\dataset2\leaf", 224, 224, 4)  # todo 修改为你的数据集的位置
    print(class_names)
    model = tf.keras.models.load_model("models/mobilenet_02.h5")  # todo 修改为训练好的mobilenet模型位置
    model.summary()
    loss, accuracy = model.evaluate(val_ds)
    print('Mobilenet test accuracy :', accuracy)
    #draw_heatmap("C:\Project\dataset2\leaf")
    #load_pickle("pickle_result.pickle")
    get_heat_map()


# 测试cnn模型的准确率
def test_cnn():
    train_ds, val_ds, class_names = data_load("C:\Project\dataset2\leaf", 224, 224, 4)  # todo 修改为你的数据集的位置
    model = tf.keras.models.load_model("models/cnn_245_epoch30.h5")  # todo 修改为训练好的cnn模型位置
    model.summary()
    loss, accuracy = model.evaluate(val_ds)
    print('CNN test accuracy :', accuracy)


# 绘制热力图，按照四个大类绘制热力图
# 注：绘制热力图这段的逻辑，因为涉及的数据比较多，为了防止出错，还是先保存为了pkl文件
def draw_heatmap(folder_name):
    # 遍历文件夹返回数目
    trash_names = ['0', '1', '2', '3', '4']
    real_label = []
    pre_label = []
    images_path = []
    folders = os.listdir(folder_name)

    for folder in folders:
        folder_path = os.path.join(folder_name, folder)
        images = os.listdir(folder_path)
        for img in images:
            xxx = folder.split("_")[0]
            x_idx = trash_names.index(xxx)
            img_path = os.path.join(folder_path, img)
            real_label.append(x_idx)
            images_path.append(img_path)


    model = tf.keras.models.load_model("models/mobilenet_02.h5")
    for ii, i_path in enumerate(images_path):
        print("{}/{}".format(ii, len(images_path) - 1))
        shutil.copy(i_path, "images/t1.jpg")
        src_i = cv2.imread("images/t1.jpg")
        src_r = cv2.resize(src_i, (224, 224))
        cv2.imwrite("images/t2.jpg", src_r)
        img = Image.open("images/t2.jpg")
        img = np.asarray(img)
        outputs = model.predict(img.reshape(1, 224, 224, 3))
        result_index = int(np.argmax(outputs))
        result = class_names[result_index]
        names = result.split("_")
        xxx = names[0]
        x_idx = trash_names.index(xxx)
        pre_label.append(x_idx)

    print(len("pre:{}".format(len(pre_label))))
    print(len("real:{}".format(len(real_label))))
    print(pre_label)
    print(real_label)
    # 先保存为pickle文件
    a_dict = {}
    a_dict["pre_label"] = pre_label
    a_dict["real_label"] = real_label
    file = open('results/pickle_result.pickle', 'wb')
    pickle.dump(a_dict, file)
    file.close()


# 加载pkl文件
def load_pickle(filename="results/pickle_result.pickle"):
    f = open("results/pickle_result.pickle", 'rb')  # 注意此处model是rb
    s = f.read()
    data = pickle.loads(s)
    # print(data)
    pre_label = data['pre_label']
    real_label = data['real_label']
    print(len(pre_label))
    print(len(real_label))
    heatmap = np.zeros((5, 5))
    for r, p in zip(real_label, pre_label):
        heatmap[r][p] = heatmap[r][p] + 1
    #heatmap = preprocessing.scale(heatmap)
    print(heatmap)
    result = []
    for row in heatmap:
        print(row)
        row = row / np.sum(row)
        result.append(row)
    return np.array(result)


# 得到热力图
def get_heat_map(array_numpy=np.random.rand(25).reshape(5, 5)):
    trash_names = ['叶片健康', '条锈病一级', '条锈病二级', '条锈病三级', '条锈病四级']
    x = array_numpy
    plt.xticks(np.arange(len(trash_names)), trash_names)
    plt.yticks(np.arange(len(trash_names)), trash_names)
    plt.imshow(x, cmap=plt.cm.hot, vmin=0, vmax=1)
    plt.title('classification heatmap')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    #test_cnn()
    test_mobilenet()
