from flask import Flask,render_template,request          # import flask
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import cv2
#from matplotlib import gridspec
#from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import utils
import tensorflow as tf

app = Flask(__name__)             # create an app instance

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'seg_op'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            print(tar_info)
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

MODEL = DeepLabModel('seg_op_500.tar.gz')

def run_visualization(model,url):
    """Inferences DeepLab model and visualizes result."""
    try:
        f = urllib.request.urlopen(url)
        jpeg_str = f.read()
        original_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('Cannot retrieve image. Please check url: ' + url)
        return

    print('running deeplab on image %s...' % url)
    
    resized_im, seg_map = model.run(original_im)
    seg_image = utils.label_to_color_image(seg_map).astype(np.uint8)
    print(seg_image.shape)
    resized_im.save('./static/resized.jpg')
    
    #cv2.imwrite("resized_im.jpg",resized_im)
    cv2.imwrite("static/seg_image.jpg",seg_image)
    #seg_image.save('./segmented.jpg')
    #return seg_image
    #print('Image saved')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']
        print('THE type of URL is',type(url))
		#data = [comment]
        run_visualization(MODEL,url)
        full_filename = os.path.join(os.getcwd(), 'seg_image.jpg')
        print('full_filename is',full_filename)
    return render_template('result.html',prediction = full_filename,url=url)
    
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == "__main__":        # on running python app.py
    app.run(host='0.0.0.0',port=5001,debug=True)                     # run the flask app


