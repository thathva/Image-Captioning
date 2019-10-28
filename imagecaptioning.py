import pickle
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array,load_img
from os import listdir
import string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import Input,LSTM,Dense,Embedding,Dropout,GRU
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu

captions='D:/Machine Learning/Project/Flicker8k_text/Flickr8k.token.txt'
images='D:/Machine Learning/Project/Flicker8k_Dataset/'

def feature_extraction(path):
    model=VGG16()
    model.layers.pop()
    model=Model(inputs=model.inputs,outputs=model.layers[-1].output)
    features=dict()
    for i in listdir(path):
        file=path + '/' + i
        img=load_img(file,target_size=(224,224))
        img=img_to_array(img)
        img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
        img=preprocess_input(img)
        ft=model.predict(img,verbose=0)
        imgid=i.split('.')[0]
        features[imgid]=ft
    return features

features=feature_extraction(images)

pickle.dump(features, open('features.pkl', 'wb'))


def doc_load(file):
    f=open(file,'r')
    text=f.read()
    f.close()
    return text

def descriptions(d_file):
    mapping_d={}
    for i in d_file.split('\n'):
        token=i.split()
        if(len(i)<2):
            continue
        imgid,imgdesc=token[0],token[1:]
        imgid=imgid.split('.')[0]
        imgdesc=' '.join(imgdesc)
        if imgid not in mapping_d:
            mapping_d[imgid]=[]
        mapping_d[imgid].append(imgdesc)
    return mapping_d

def preprocess_text(desc):
    t=str.maketrans('','',string.punctuation)
    for i,j in desc.items():
        for k in range(len(j)):
            de=j[k]
            de=de.split()
            de=[word.lower() for word in de]
            de=[w.translate(t) for w in de]
            de=[word for word in de if len(word)>1]
            de=[word for word in de if word.isalpha()]
            j[k]=' '.join(de)
            
def vocabulary(descrip):
    alldesc=set()
    for i in descrip.keys():
        [alldesc.update(d.split()) for d in descrip[i]]
    return alldesc
            
def savedescriptions(description,file):
    lines=[]
    for i,j in description.items():
        for k in j:
            lines.append(i+' '+k)
    data='\n'.join(lines)
    f=open(file,'w')
    f.write(data)
    f.close()
    
document=doc_load(captions)
description=descriptions(document)
preprocess_text(description)
vocab=vocabulary(description)
savedescriptions(description,'description.txt')


def set_load(file):
    doc=doc_load(file)
    dataset=[]
    for i in doc.split('\n'):
        if(len(i))<1:
            continue
        ident=i.split('.')[0]
        dataset.append(ident)
    return set(dataset)

def load_clean_descriptions(filename, dataset):
	# load document
	doc = doc_load(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

def image_features(file,data):
    feats=pickle.load(open(file,'rb'))
    feature={k:feats[k] for k in data}
    return feature


trainfile='D:/Machine Learning/Project/Flicker8k_text/Flickr_8k.trainImages.txt'
train=set_load(trainfile)
train_description=load_clean_descriptions('description.txt',train)
train_features=image_features('features.pkl',train)        

def linestext(description):
    alldesc=[]
    for key in description.keys():
        [alldesc.append(d) for d in description[key]]
    return alldesc

def tokenize(desc):
    line=linestext(desc)
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(line)
    return tokenizer

tokenizer=tokenize(train_description)
vocab_size=len(tokenizer.word_index)+1
'''
def sequence(tokenizer,maxlen,description,images):
    X1,X2,y=[],[],[]
    for i,j in description.items():
        for k in j:
            seq=tokenizer.texts_to_sequences([k])[0]
            for ii in range(1,len(seq)):
                inseq,outseq=seq[:ii],seq[ii]
                inseq=pad_sequences([inseq],maxlen=maxlen)[0]
                outseq=to_categorical([outseq],num_classes=vocab_size)[0]
                X1.append(images[i])[0]
                X2.append(inseq)
                y.append(outseq)
    return np.array(X1),np.array(X2),np.array(y)
'''
def sequence(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return np.array(X1), np.array(X2), np.array(y)

def max_length(descriptions):
	lines = linestext(descriptions)
	return max(len(d.split()) for d in lines)


def model_defn(vocab_size,max_length):
    inputs1=Input(shape=(4096,))
    fe1=Dropout(0.5)(inputs1)
    fe2=Dense(256,activation='relu')(fe1)
    inputs2=Input(shape=(max_length,))
    se1=Embedding(vocab_size,256,mask_zero=True)(inputs2)
    se2=Dropout(0.5)(se1)
    se3=GRU(256)(se2)
    decoder1=add([fe2,se3])
    decoder2=Dense(256,activation='relu')(decoder1)
    outputs=Dense(vocab_size,activation='softmax')(decoder2)
    model=Model(inputs=[inputs1,inputs2],outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    return model


def data_generator(descriptions, photos, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = sequence(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]
            
            

max_length=max_length(train_description)
X1train,X2train,ytrain=sequence(tokenizer,max_length,train_description,train_features)

valfile='D:/Machine Learning/Project/Flicker8k_text/Flickr_8k.devImages.txt'
test = set_load(valfile)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('description.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = image_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = sequence(tokenizer, max_length, test_descriptions, test_features)



model = model_defn(vocab_size, max_length)
epochs = 20
steps = len(train_description)
for i in range(epochs):
	# create the data generator
	generator = data_generator(train_description, train_features, tokenizer, max_length)
	# fit for one epoch
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps,verbose=1)
	# save model
	model.save('model_' + str(i) + '.h5')


#model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))

# data generator, intended to be used in a call to model.fit_generator()

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

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc_list in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    
filename = 'D:/Machine Learning/Project/Flicker8k_text/Flickr_8k.testImages.txt'
test = set_load(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('description.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = image_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))


from keras.utils import plot_model
plot_model(model, to_file='model.png')

# load the model
filename = 'model_19.h5'
model = load_model(filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)


# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature



pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
p='pic.jpg'
photo = extract_features('D:/Machine Learning/Project/e.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, 34)
print(description)

'''
with LSTM
BLEU-1: 0.502306
BLEU-2: 0.265127
BLEU-3: 0.176106
BLEU-4: 0.076345

with GRU
BLEU-1: 0.537460
BLEU-2: 0.283245
BLEU-3: 0.191800
BLEU-4: 0.088605
'''


from google.cloud import texttospeech

# Instantiates a client
client = texttospeech.TextToSpeechClient()

# Set the text input to be synthesized
synthesis_input = texttospeech.types.SynthesisInput(text=description)

# Build the voice request, select the language code ("en-US") and the ssml
# voice gender ("neutral")
voice = texttospeech.types.VoiceSelectionParams(
    language_code='en-US',
    ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

# Select the type of audio file you want returned
audio_config = texttospeech.types.AudioConfig(
    audio_encoding=texttospeech.enums.AudioEncoding.MP3)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(synthesis_input, voice, audio_config)

# The response's audio_content is binary.
with open('output.mp3', 'wb') as out:
    # Write the response to the output file.
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')
