# Video Game Rating Predictor

## Overview

This project aims to build a model can be used to predict ESRB (Entertainment Software Rating Board) ratings for video games basing on their content descriptors.

It's able to classify games into one of the following age groups:

- Everyone
- Everyone 10+
- Teen
- Mature

For more information about ESRB rating categories, refer to the [ESRB Ratings Guide](https://www.esrb.org/ratings-guide/).

## Model Usage Example

Script:

```python
import pickle
import numpy as np

# Load the ESRB model from the pickle file
with open('esrb-model.pkl', 'rb') as file:
    esrb_model = pickle.load(file)

# Example game data
Minecraft = np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

# Predict ESRB rating
def predict_esrb_rating(game_data):
    predicted_rating = esrb_model.predict(game_data.reshape(1, -1))
    if predicted_rating[0] == 0:
        return "Everyone"
    elif predicted_rating[0] == 1:
        return "Everyone 10+"
    elif predicted_rating[0] == 2:
        return "Teen"
    elif predicted_rating[0] == 3:
        return "Mature 17+"

print(predict_esrb_rating(Minecraft))
```

Output:

```
Everyone 10+
```
