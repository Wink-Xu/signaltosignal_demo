import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import pandas as pd 
import os 
import tensorflow as tf


def load_data(data_dir):
    data = np.array(pd.read_csv(data_dir))
    return data[:,1:]

def save_data(x,data_dir):
    save_data = pd.DataFrame(x)
    save_data.to_csv(data_dir)
	
def save_model(sess,ckpt_dir,global_step,model_name='signaltosignal'):
    saver = tf.train.Saver()
    checkpoint_dir = ckpt_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("Saving model....")
    saver.save(sess,os.path.join(checkpoint_dir,model_name),global_step = global_step)

def load_model(ckpt_dir):
    print("Loading model...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        full_path = tf.train.latest_checkpoint(ckpt_dir)
        global_step = int(full_path.split('/')[-1].split('-')[-1])
        saver.restore(sess,full_path)
        return True,global_step
    else :
        return False,0
		

def save_image(file_name,original_image,generate_image,noise_image):
    original_image = normalize(np.squeeze(original_image))
    generate_image = normalize(np.squeeze(generate_image))
    noise_image = normalize(np.squeeze(noise_image))
    
    image = np.concatenate([original_image,generate_image,noise_image],axis = 1)
    im = Image.fromarray(image.astype('uint8')).convert('L')
    im.save(file_name, 'png')

def normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))*255   