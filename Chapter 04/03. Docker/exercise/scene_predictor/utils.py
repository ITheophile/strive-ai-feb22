import os
from flask import request, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploaded_images'

def store_uploaded_image():
    '''
    Stores the uploaded images into the UPLOAD_FOLDER 
    within the static folder
    '''
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_name = secure_filename(file.filename)
            file_path = os.path.join('static', UPLOAD_FOLDER, file_name)
            file.save(file_path)
            return file_path
        else:
            return None




def clean_upload_folder():
    """
    remove images from UPLOAD_FOLDER
    """
    upload_folder_path = os.path.join('static', UPLOAD_FOLDER)
    image_names = os.listdir(upload_folder_path)
    for image in image_names:
        img_pth = os.path.join(upload_folder_path, image)
        os.remove(img_pth)

