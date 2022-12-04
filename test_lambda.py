import requests 

#url = "http://localhost:8080/2015-03-31/functions/function/invocations"
url = "https://129j0j70wa.execute-api.us-east-2.amazonaws.com/test"
data = {'url': 'http://bit.ly/mlbookcamp-pants'}

result = requests.post(url=f"{url}/predict-clothing", json=data).json()
print(result)
