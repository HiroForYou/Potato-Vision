from flask import Flask, render_template, request
from prediction import detect_and_classification
import os
import base64
from io import BytesIO
            
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
   return render_template('start.html')

@app.route('/prediction',methods=["GET","POST"])
def prediction():
    errors = []
    allowed_extensions = set(['png', 'jpg', 'jpeg', 'gif'])
    
    try:
        if request.method == 'POST':
            currentfile = request.files.get('file', '')
    except:
        errors.append(
                    "Unable to read file. Please make sure it's valid and try again."
                    )
    # prediction of model
    image_result, quantity, quality = detect_and_classification(currentfile, threshold = .5)    
          
    buffered = BytesIO()
    image_result = image_result.resize((800, 600))
    image_result.save(buffered, format="JPEG")
    image_memory = base64.b64encode(buffered.getvalue())

    return render_template("result.html", quantity=quantity, quality=quality, img_data=image_memory.decode('utf-8'))


if __name__ == '__main__':
    
    app.run(debug=True)
