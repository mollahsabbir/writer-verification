from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from PIL import Image
import os

from writerverifier import WriterVerifier

app = Flask(__name__)
verifier = WriterVerifier("saved_models/random.pth")

# TODO: Change source of secret in production
app.config['SECRET_KEY'] = 'onlyfordevelopment'

class ImageUploadForm(FlaskForm):
    image_1 = FileField("First Image")
    image_2 = FileField("Second Image")
    submit = SubmitField("Check Similarity")

@app.route('/', methods=['GET', 'POST'])
def home():
    form = ImageUploadForm()
    score = None
    if form.validate_on_submit():
        image_1 = Image.open(request.files['image_1'])
        image_2 = Image.open(request.files['image_2'])
        verifier.get_embedding(image_1)
        score = verifier.get_score(image_1, image_2)
        
    return render_template("index.html", form=form, score=score)

if __name__=="__main__":
    app.run(debug=True)

