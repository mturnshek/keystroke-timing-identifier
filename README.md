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

You will be asked your name. This name will be used as a class in the classifier.

It is assumed that all data points collected under a name are for the same typing style.

When collecting a data point, the user will be prompted with a random food review from Amazon.

This can be used to bypass typer's block.

#### Train model
`python3 train.py`

#### Test inference
`sudo python3 infer.py`

You will be prompted with a food review. The situation is the same as in the data collection phase.

After a single collected data point, the model will predict who you are and output the name.

## Notes

#### Current model architecture
- one 1D conv layer
- one hidden layer
- softmax with num_classes

##### Larger dataset collection
Collecting enough data from lots of different users is important. Releasing a public frontend seems best for easy data collection.

It will be hard to get people to type enough.

It's almost easier to get people to talk with one another while using it and collect that data. How can lots of people to talk with one another, and have fun while knowing their data is being collected for a project? Online turing test where one is paired with either a SOTA language model or another human?