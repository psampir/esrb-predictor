from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

class CustomCORS(CORS):
    def after_request(self, response):
        response = super().after_request(response)
        response.headers.add('Access-Control-Allow-Private-Network', 'true')
        return response

app = Flask(__name__)
CORS = CustomCORS(app, resources={r"/predict": {"origins": "*"}})

with open('esrb-model.pkl', 'rb') as file:
    esrb_model = pickle.load(file)

def predict_esrb_rating(game_data):
    predicted_rating = esrb_model.predict(game_data.reshape(1, -1))
    if predicted_rating[0] == 0:
        return "Everyone"
    elif predicted_rating[0] == 1:
        return "Everyone 10+"
    elif predicted_rating[0] == 2:
        return "Teen"
    else:
        return "Mature 17+"

# Define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    game_data = np.array(data['game_data'])
    rating = predict_esrb_rating(game_data)
    return jsonify({'rating': rating})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5001)
