from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_placement():
    # oncgpa = float(request.form.get('cgpa'))
    # iq = int(request.form.get('iq'))
    # profile_score = int(request.form.get('profile_score'))
    file = request.files['image']
    folder='C:/Users/Dell Gaming/Desktop/project/DL/images/'
    filepath = os.path.join(folder, file.filename)
    file.save(filepath)
    #print(filepath)
    img = load_and_prep_image(filepath)
    result = model.predict(tf.expand_dims(img, axis=0))
    class_names = ['apple_pie', 'baby_back_ribs', 
                   'baklava', 'beef_carpaccio', 'beef_tartare',
                    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 
                    'breakfast_burrito', 'bruschetta', 'caesar_salad', 
                    'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 
                    'cheesecake', 'cheese_plate', 'chicken_curry', 
                    'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 
                    'chocolate_mousse', 'churros', 'clam_chowder', 
                    'club_sandwich', 'crab_cakes', 'creme_brulee', 
                    'croque_madame', 'cup_cakes', 'deviled_eggs',
                      'donuts', 'dumplings', 'edamame', 'eggs_benedict', 
                      'escargots', 'falafel', 'filet_mignon', 
                      'fish_and_chips', 'foie_gras', 'french_fries', 
                      'french_onion_soup', 'french_toast', 'fried_calamari', 
                      'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi'
                      , 'greek_salad', 'grilled_cheese_sandwich',
                        'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 
                        'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 
                        'hummus', 'ice_cream', 'lasagna', 'lobster_bisque',
                          'lobster_roll_sandwich', 'macaroni_and_cheese', 
                          'macarons', 'miso_soup', 'mussels', 'nachos', 
                          'omelette', 'onion_rings', 'oysters', 'pad_thai', 
                          'paella', 'pancakes', 'panna_cotta', 'peking_duck',
                            'pho', 'pizza', 'pork_chop', 'poutine', 
                            'prime_rib', 'pulled_pork_sandwich', 'ramen', 
                            'ravioli', 'red_velvet_cake', 'risotto', 
                            'samosa', 'sashimi', 'scallops', 'seaweed_salad', 
                            'shrimp_and_grits', 'spaghetti_bolognese', 
                            'spaghetti_carbonara', 'spring_rolls', 'steak', 
                            'strawberry_shortcake', 'sushi', 'tacos', 
                            'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
    print(len(class_names))
    pred_class = class_names[result.argmax()]
    print(pred_class)
    return jsonify(message=pred_class)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)

def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.io.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  
  return img