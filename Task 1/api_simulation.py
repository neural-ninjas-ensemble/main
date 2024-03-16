import requests
def model_stealing(path_to_png_file: str):
    SERVER_URL = "[paste server url here]"
    ENDPOINT = "/modelstealing"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "[paste your team token here]"

    with open(path_to_png_file, "rb") as img_file:
        response = requests.get(
            URL, files={"file": img_file}, headers={"token": TEAM_TOKEN}
        )

        if response.status_code == 200:
            return response.content["representation"]
        else:
            raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")
        
def model_stealing_reset():
    SERVER_URL = "[paste server url here]"
    ENDPOINT = "/modelstealing/reset"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "[paste your team token here]"

    response = requests.post(
        URL, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        print("Endpoint rested successfully")
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")
    
    
    
    #ładownaie danych 
    
    
    import torch
from taskdataset import TaskDataset


if __name__ == "__main__":
    dataset = torch.load("modelstealing/data/ExampleModelStealingPub.pt")

    print(dataset.ids, dataset.imgs, dataset.labels)
    
    
    
def model_stealing_submission(path_to_onnx_file: str):
    SERVER_URL = "[paste server url here]"
    ENDPOINT = "/modelstealing/submit"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "[paste your team token here]"

    with open(path_to_onnx_file, "rb") as onnx_file:
        response = requests.post(
            URL, files={"file": onnx_file}, headers={"token": TEAM_TOKEN}
        )

        if response.status_code == 200:
            return response.content["score"]
        else:
            raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")