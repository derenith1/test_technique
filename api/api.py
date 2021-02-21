import flask
from flask import request

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network import return_prediction


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/predict', methods=['GET'])
def prediction():
    if 'height' in request.args:
        height = float(request.args['height'])
    else:
        return "No height provided"

    if 'width' in request.args:
        width = float(request.args['width'])
    else:
        return "No width provided"

    if 'depth' in request.args:
        depth = float(request.args['depth'])
    else:
        return "No depth provided"

    if 'weight' in request.args:
        weight = float(request.args['weight'])
    else:
        return "No weight provided"

    data = [height, width, depth, weight]
    predict = {
        "activity": return_prediction(data)
    }
    return predict


app.run()