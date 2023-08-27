from boctor.train import train
from boctor.predict import  predict


file_path = '/home/ubuntu/Desktop/Projects/Boctor/boctor/dataset/intents.json'

model_dir = '/home/ubuntu/Desktop/Projects/Boctor/boctor/saved_models/'
epochs = 10
learning_rate = 0.05

model=train(file_path, epochs, learning_rate,model_dir)

text = "what is your name?"

response = predict(text,model_dir,file_path)

print(response)

#you can use a flask server and frontend to make a chatbot . 



