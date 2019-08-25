import pandas as pd
import numpy as np
import random


class FoodReviewGenerator:
    def __init__(self):
        self.reviews = pd.read_csv('./food_reviews/Reviews.csv')
        self.length = 568453

    def random(self):
        generation = ""
        while len(generation) < 160:
            i = random.randrange(0, self.length)
            generation = self.reviews['Text'][i]
        return generation

if __name__ == '__main__':
    # test
    generator = FoodReviewGenerator()
    print(generator.random())