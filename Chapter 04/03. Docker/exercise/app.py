from utils import store_uploaded_image, clean_upload_folder
from flask import Flask, render_template


app = Flask(__name__)



@app.route('/', methods = ['GET','POST'])
def index():
    
    output = store_uploaded_image()
    if output:
        
        return render_template('test.html', test = output)
    return render_template('index.html')

print(clean_upload_folder())
    

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)