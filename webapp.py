from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os

from modelserver import get_score

app = Flask(__name__)

# TODO: Change source of secret in production
app.config['SECRET_KEY'] = 'onlyfordevelopment'
app.config['UPLOAD_FOLDER'] = 'static/files'

class ImageUploadForm(FlaskForm):
    image_1 = FileField("First Image")
    image_2 = FileField("Second Image")
    submit = SubmitField("Check Similarity")

@app.route('/', methods=['GET', 'POST'])
def home():
    form = ImageUploadForm()
    score = None
    if form.validate_on_submit():
        image_1 = form.image_1.data
        image_2 = form.image_2.data
        score = get_score(image_1, image_2)
        
    return render_template("index.html", form=form, score=score)

if __name__=="__main__":
    app.run(debug=True)

