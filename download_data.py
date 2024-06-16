import requests 

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

with open('data/input.txt', 'w') as file:
    file.write(response.text)

print('Data downloaded successfully')