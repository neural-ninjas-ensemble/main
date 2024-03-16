'''
Functions provided by the organizers to query the model.
'''
import requests

from typing import List

from config import SERVER_URL, TOKEN


def sybil_attack(ids: List[int], home_or_defense: str, binary_or_affine: str):
    if home_or_defense not in ["home", "defense"] or binary_or_affine not in ["binary", "affine"]:
        raise "Invalid endpoint"
    
    ENDPOINT = f"/sybil/{binary_or_affine}/{home_or_defense}"
    URL = SERVER_URL + ENDPOINT
    
    TEAM_TOKEN = TOKEN
    ids = ids = ",".join(map(str, ids))

    response = requests.get(
        URL, params={"ids": ids}, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        return response.content["representations"]
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")


def sybil_attack_reset():
    ENDPOINT = "/sybil/reset"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = TOKEN

    response = requests.post(
        URL, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        print("Endpoint rested successfully")
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")


if __name__ == '__main__':
    sybil_attack_reset()