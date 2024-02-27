# Importing required libs
from flask import Flask
from flask import render_template, request
from model import preprocess_img, predict_result
import cv2

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    # try:
    if request.method == 'POST':
        # img = preprocess_img(request.files['file'].stream)
        img = cv2.imread(request.files['file'].stream)
        _, pred = predict_result(img)
        result_string = ', '.join(pred)
        print("result_string: ", result_string)
        return render_template("result.html", predictions=result_string)

    # except:
    #     error = "File cannot be processed."
    #     return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
