from flask import Flask, render_template
import logging

logger = logging.getLogger("ma")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

