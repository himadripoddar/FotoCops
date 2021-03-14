from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def photo_detector():
    if request.method == 'POST':
        image_url = request.form['fname']

        return render_template("index.html", image_url=image_url)
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
