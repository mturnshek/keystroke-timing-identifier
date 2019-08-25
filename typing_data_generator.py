import os
from time import time as now
from time import sleep

import keyboard
import numpy as np

from food_review_generator import FoodReviewGenerator


class TypingDataGenerator:
    def __init__(self):
        self.warmup = 10
        self.cooldown = 10
        self.slice = 128
        self.food_review_generator = FoodReviewGenerator()
    
    def get_one_datapoint(self):
        prev_time = now()
        recording = keyboard.start_recording()

        while True:
            sleep(0.01)

            # progress bar implementation in the middle of this code...
            progress = ''.join(['=']*len(list(recording[0].queue)))
            empty = ''.join([' ']*(self.warmup + self.slice + self.cooldown - len(list(recording[0].queue))))
            print('\r' + '[' + progress + empty + ']', end='')

            if (len(list(recording[0].queue)) >= self.warmup + self.slice + self.cooldown):
                recorded = keyboard.stop_recording()
                break

        times = []
        for k in recorded:
            times.append(k.time - prev_time)
            prev_time = k.time

        X = np.array(times)
        X = X[self.warmup:]
        X = X[:-self.cooldown]
        X = X[:self.slice]
        X = np.reshape(X, (self.slice, 1))

        return X

    def y_to_name(self, y):
        return os.listdir('data')[np.argmax(y)]


if __name__ == '__main__':
    name = input('What is your name? > ')

    # folder where that name's data is stored
    folder = f'data/{name}'

    # create the folder if the name has not been seen
    if name not in os.listdir('data'):
        os.mkdir(folder)
        os.mkdir(folder + '/X')

    food_review_generator = FoodReviewGenerator()
    typing_data_generator = TypingDataGenerator()

    # continually prompt the user to type, collect their keystroke timing data
    while True:
        input('\nPress enter to continue to the next sample, or ctrl+C to quit.')
        print('\n===\n')
        print(food_review_generator.random(), '\n')

        filenames = os.listdir(folder + '/X')
        file_nos = map(lambda filename: filename.split('.')[0], filenames)
        if filenames == []:
            file_no = str(0).zfill(5)
        else:
            file_no = str(int(max(file_nos)) + 1).zfill(5)

        x = typing_data_generator.get_one_datapoint()

        np.save(f'{folder}/X/{file_no}.npy', x)