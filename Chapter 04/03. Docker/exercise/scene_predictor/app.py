from utils import store_uploaded_image, clean_upload_folder
from model.model_setup import model
from model.prediction import make_prediction

import os
from torch import load
from PIL import Image
from flask import Flask, render_template


app = Flask(__name__)



@app.route('/', methods = ['GET','POST'])
def index():
    
    # start by removing any previous image 
    # -> no need to accumulate previously loaded images
    clean_upload_folder()

    # Store the actual image and return its path
    img_pth = store_uploaded_image()

    # test output
    if img_pth:
        
        # load image
        img = Image.open(img_pth)

        # Load prediction classes
        classes = load(os.path.join('model','classes.pth'))

        # Load model with trained weights
        model.load_state_dict( 
            load( os.path.join('model','model_state.pth') ) 
        )

        
    
        pred = make_prediction(model, img, classes)


        # image path for displaying in the bootstrap card
        if img_pth.find('\\') > 0: # backslash found
            new_file_path = img_pth.replace('\\', '/').replace('static/', '')
        else: # forward slash
            new_file_path = img_pth.replace('static/', '')

        return render_template('index.html', prediction = pred.upper(), img_pth = new_file_path)

    else:
        return render_template('index.html')
        

    
    
# clean_upload_folder()



if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)