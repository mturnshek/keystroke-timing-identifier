from model import Identifier


if __name__ == '__main__':
    identifier_model = Identifier()
    identifier_model.train(epochs=500)
    identifier_model.save('identifier_model')