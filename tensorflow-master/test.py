# -*- coding: UTF-8 –*-
import tensorflow as tf
import base64
import sys
import cv2
from PIL import Image
import numpy as np
import shutil
from flask import Flask, request, make_response, json, render_template

app = Flask(__name__)

#"images/hhh.JPG"
class preImg():
    def __init__(self):                  # 添加初始化函数
        self.model = tf.keras.models.load_model("models/mobilenet_02.h5")  # todo 修改为自己的模型路径
        self.to_predict_name ="images/hhh.JPG"
        self.class_names = ['leaf_healthy', 'leaf_rust_1', 'leaf_rust_2', 'leaf_rust_3', 'leaf_rust_4']

    def work(self):
            img_name = self.to_predict_name
            target_image_name = "images/tmp_single" + img_name.split(".")[-1]
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
            img_init = cv2.resize(img_init, (224, 224))
            cv2.imwrite('images/target.png', img_init)


    # 预测图片
    def predict_img(self):
        img = Image.open('images/target.png')
        img = np.asarray(img)
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))
        result_index = int(np.argmax(outputs))
        result = self.class_names[result_index]
        names = result.split("_")
        return(result)

@app.route('/check' ,methods=['GET','POST'])
def check():
    if request.method == 'POST':
        return "sb"
#        src = request.form.get('bs64')
#        data = src.split(',')[1]
#        image_data = base64.b64decode(data)
#        with open('temp.gif', 'wb') as f:
#            f.write(image_data)
#        x.to_predict_name ="temp.gif"
#        x.work()
#        return x.predict_img()


@app.route('/')
def index():           # index为视图函数
    return render_template("Home.html")








if __name__ == '__main__':
    x = preImg()
    app.run()




# x.work()
# x.predict_img()
#
# x.to_predict_name="images/123.jpg"
# x.work()
# x.predict_img()



