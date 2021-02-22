import flask
from flask import request

import os, sys

# Add the path of the project directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network import return_prediction

app = flask.Flask(__name__)
app.config["DEBUG"] = True


# Route for prediction : /predict, method : GET to retrieve the infos

@app.route('/predict', methods=['GET'])
def prediction():
    """
        Returns the prediction of the network calling the return_prediction function.
        This function retrieve via the GET method the height, width, depth and weight values provided by the user.

                Returns:
                        predict (dict): Dictionary of the activity predicted. Either 'mlp', 'deco' or 'meuble'.
                        Raise an error in the return_predict function if the dimension are abnormal.
        """

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

    # Collect the data in a list to feed the predictor
    data = [height, width, depth, weight]
    predict = {
        "activity": return_prediction(data)
    }
    return predict


# run the API
app.run()
