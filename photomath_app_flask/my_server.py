import flask
import numpy
import imageio
import os
#import werkzeug

from predict_on_image import *
from string_eval_expression import *




app = flask.Flask(__name__)
@app.route('/', methods = ['GET', 'POST'])
def welcome():
    return "Send a picture to this URL and get prediction for your handwritten math expression :)"
@app.route('/predict/', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image0']
    filename = imagefile.filename
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
    
    put_model = "my_model_good_final.h5"
    pred, sol = do_prediction(imagefile.filename, put_model)
    
    return str(pred)+"\n"+str(sol)
app.run(host="0.0.0.0", port=os.environ.get('PORT', 5000), debug=True)
