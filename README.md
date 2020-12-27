## Goal
Identify a person by their keystrokes timing.

## Usage

#### Dependencies
* python3
* tensorflow 2
* pandas
* numpy
* keyboard

#### Download prompts dataset
`bash download_food_reviews.sh`

#### Collect data
`sudo python3 typing_data_generator.py`

You will be asked your name. This name will be used as a class in the classifier. It is assumed that all data points collected under a name are for the same typing style.

When collecting a data point, the user will be prompted with a random food review from Amazon. This can be used to bypass typer's block.

#### Train model
`python3 train.py`

#### Test inference
`sudo python3 infer.py`

You will be prompted with a food review, like in the data collection phase. After a single collected data point, the model will predict who you are and output the name.

## Notes

#### Current model architecture
- one 1D conv layer
- one hidden layer
- softmax with num_classes
