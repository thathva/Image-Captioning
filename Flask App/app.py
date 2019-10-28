from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from flask import Flask,request,jsonify
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing.image import img_to_array,load_img
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
import base64
from PIL import Image
import scipy
import io

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
global graph

graph = tf.get_default_graph()

def load():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    filename = 'model_19.h5'
    model = load_model(filename)
    

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))


def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = np.argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

def extract_features(image):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # load the photo
    print(image.shape)
    #image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    #image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    print(image.shape)
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature

def preprocess_image(image,target_size):
    if image.mode!="RGB":
        image=image.convert("RGB")
    image=image.resize(target_size)
    image=img_to_array(image)
   # image=np.expand_dims(image,axis=0)

    return image

#photo = extract_features('D:/Machine Learning/Project/example.jpg')
@app.route('/',methods=['POST'])
def predict():
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    with graph.as_default(): 
        image=Image.open(io.BytesIO(decoded))
        processed_image=preprocess_image(image,target_size=(224,224))
        im=extract_features(processed_image)
        prediction=generate_desc(model, tokenizer, im, 34)
    
        response={
            'prediction': {
                'person':prediction
                
            }
    }
    
    return jsonify(response)
        #description = generate_desc(model, tokenizer, photo, 34)
        #return description[8:-7]

if __name__ == '__main__':
    load()
    app.run(port=5000,debug=True)
