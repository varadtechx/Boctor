import random
import json
import os 
import torch

from boctor.model import NeuralNet
from boctor.utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(sentence, model_dir,file_path):
    with open(file_path, 'r') as json_data:
        intents = json.load(json_data)

    FILE = os.path.join(model_dir,"model.pth")
    if not os.path.isfile(FILE):
        raise Exception("Save(Train)/Rename  model named as model.pth in saved_model folder if not present")
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    
    if sentence is None:
        return "Please enter a sentence"

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."