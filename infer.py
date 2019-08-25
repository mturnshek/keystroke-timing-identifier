import tensorflow as tf
import numpy as np

from typing_data_generator import TypingDataGenerator
from food_review_generator import FoodReviewGenerator
from model import Identifier


if __name__ == '__main__':
  # load generators and model
  food_review_generator = FoodReviewGenerator()
  typing_data_generator = TypingDataGenerator()
  identifier_model = tf.keras.models.load_model('identifier_model')

  # collect user input once
  print('\n', food_review_generator.random(), '\n')
  data_point = typing_data_generator.get_one_datapoint()

  # forward pass, match with a name, get confidence of the result
  result = identifier_model.predict(np.array([data_point]))
  name = typing_data_generator.y_to_name(result[0])
  confidence = np.max(result) # this is not exactly correct, but it is relatively correct

  print('\n', name, confidence, '\n')
