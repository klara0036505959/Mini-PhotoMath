import cv2
import numpy as np
import itertools
import PIL
import tensorflow as tf
from PIL import Image
from photomath_app_flask.string_eval_expression import *
#from preprocess import rescale_segment as rescale_segment
#from preprocess import extract_segments as extract_segments
def rescale_segment( segment, size = [28,28], pad = 0 ):
    '''function for resizing (scaling down) images
    input parameters
    seg : the segment of image (np.array)
    size : out size (list of two integers)
    output 
    scaled down image'''
    if len(segment.shape) == 3 : # Non Binary Image
        import cv2
        # thresholding the image
        ret,segment = cv2.threshold(segment,127,255,cv2.THRESH_BINARY)
    m,n = segment.shape
    idx1 = list(range(0,m, (m)//(size[0]) ) )
    idx2 = list(range(0,n, n//(size[1]) )) 
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i,j] = segment[ idx1[i] + (m%size[0])//2, idx2[j] + (n%size[0])//2]
    return out

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = (target_length - array.shape[axis]) // 2
    remaining = (target_length - array.shape[axis]) % 2

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (pad_size , pad_size + remaining)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


def extract_segments(img, pad=30, reshape = 0,size = [28,28], area = 150, threshold = 100, 
                     gray = False, dil = True, ker = 1, squared = 1) :
    
    import cv2
    
    # thresholding the image
    ret,thresh1 = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    
    # Negative tranform gray levels (background becomes black) 
    thresh1 = 255 - thresh1
    img = 255 - img

    # connected component labelling 
    output = cv2.connectedComponentsWithStats(thresh1, 4)
    final = []
    cords_of_segments = []
    temp2 = output[2]

    #ova linija mozda ne treba
    temp2 = temp2[temp2[:,4]>area]

    temp1 = np.sort( temp2[:,0] )
    kernel = np.ones( [ker, ker])

    for i in range(1,temp2.shape[0]):
        cord = np.squeeze( temp2[temp2[:,0] == temp1[i]] )
#         import pdb; pdb.set_trace()
#         print(cord)
    
        if gray == False:
            num = np.pad( thresh1[ cord[1]:cord[1]+cord[3], cord[0]:cord[0]+cord[2] ], pad,'constant')
        else :
            num = np.pad( img[ cord[1]:cord[1]+cord[3], cord[0]:cord[0]+cord[2] ], pad,'constant')

        if dil :
            num = cv2.dilate(num,kernel,iterations = 1)
        else :
            num = cv2.erode(num,kernel,iterations = 1)

        if reshape == 1:
            num = rescale_segment( num, size )
        if squared == 1:
            x,y = num.shape[0], num.shape[1]
            desired = max(x,y)
            axis = 0 if x < y else 1
            num = pad_along_axis(num, desired, axis)
        final.append(num/255)
        #x,y,w,h
        cords_of_segments.append((cord[0], cord[1], cord[2], cord[3]))
        
    return final, cords_of_segments


def do_prediction(putanja_slike, putanja_model):
    class_names = ['/', '(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '*']
    model = tf.keras.models.load_model(putanja_model)

    im1 = cv2.imread(putanja_slike,0)
    kernel = np.ones( [3,3])
    im1 = cv2.erode(im1,kernel,iterations = 1)
    segments, cords_of_segments = extract_segments(im1, pad = 5, reshape = 0, size = [28,28], 
                               threshold = 40, area = 800, ker = 1, gray = True, squared = 1)
     
    expressions = []
    final_predictions = ""
    for j in range(len(segments)):
        img = Image.fromarray((segments[j]*255).astype(np.uint8))
        img = img.resize(size=(28, 28))
        img_np = np.array(img)
        img_np = img_np /255
        img_array = np.expand_dims(img_np, axis=0)
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_array = tf.expand_dims(img_array, axis=-1)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        final_predictions += str(class_names[np.argmax(score)]) + " "
        
    final_predictions = final_predictions[:-1]
    
    string = fix_neg_numbers(final_predictions)
    string, prev = fix_multidigit_numbers(string)
    if (string != prev): print("Fixed for multidigit numbers: " + string)
    obj = Conversion(len(string))
    string = obj.infixToPostfix(string)
    try:
      obj = evalpostfix()
      sol = obj.centralfunc(string)
      return final_predictions, sol
    except:
      return final_predictions, ""
    
   

