from typing import List
from dotenv import load_dotenv

import numpy as np
import requests
import os

load_dotenv()

SERVER_URL = "http://34.71.138.79:9090"
TEAM_TOKEN = os.getenv("TEAM_TOKEN")


def sybil(ids: List[int], home_or_defense: str, binary_or_affine: str):
    if home_or_defense not in ["home", "defense"] or binary_or_affine not in [
        "binary",
        "affine",
    ]:
        raise Exception("Invalid endpoint")

    endpoint = f"/sybil/{binary_or_affine}/{home_or_defense}"
    url = SERVER_URL + endpoint
    ids = ",".join(map(str, ids))
    response = requests.get(url, params={"ids": ids}, headers={"token": TEAM_TOKEN})
    if response.status_code == 200:
        representations = response.json()["representations"]
        ids = response.json()["ids"]
        return representations
    else:
        raise Exception(
            f"Sybil failed. Code: {response.status_code}, content: {response.json()}"
        )


# Be careful. This can be done only 4 times an hour.
# Make sure your file has proper content.
def sybil_submit(binary_or_affine: str, path_to_npz_file: str):
    if binary_or_affine not in ["binary", "affine"]:
        raise Exception("Invalid endpoint")

    endpoint = f"/sybil/{binary_or_affine}/submit"
    url = SERVER_URL + endpoint

    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})

    if response.status_code == 200:
        print("OK")
        print(response.json())
    else:
        print(
            f"Request submit failed. Status code: {response.status_code}, content: {response.json()}"
        )


def sybil_reset(binary_or_affine: str, home_or_defense: str):
    if binary_or_affine not in ["binary", "affine"]:
        raise Exception("Invalid endpoint")
    
    if home_or_defense not in ["home", "defense"]:
        raise Exception("Invalid endpoint")

    endpoint = f"/sybil/{binary_or_affine}/reset/{home_or_defense}"
    url = SERVER_URL + endpoint
    response = requests.post(url, headers={"token": TEAM_TOKEN})
    if response.status_code == 200:
        print("Request ok")
        print(response.json())
    else:
        raise Exception(
            f"Sybil reset failed. Code: {response.status_code}, content: {response.json()}"
        )


# Be careful. This can be done only once an hour.
# Computing this might take a few minutes. Be patient.
# Make sure your file has proper content.
def defense_submit(path_to_npz_file: str):
    endpoint = "/defense/submit"
    url = SERVER_URL + endpoint
    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            print("Request ok")
            print(response.json())
        else:
            raise Exception(
                f"Defense submit failed. Code: {response.status_code}, content: {response.json()}"
            )


def model_stealing(path_to_png_file: str):
    endpoint = "/modelstealing"
    url = SERVER_URL + endpoint
    with open(path_to_png_file, "rb") as f:
        response = requests.get(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            representation = response.json()["representation"]
            return representation
        else:
            raise Exception(
                f"Model stealing failed. Code: {response.status_code}, content: {response.json()}"
            )


def model_stealing_submit(path_to_onnx_file: str):
    endpoint = "/modelstealing/submit"
    url = SERVER_URL + endpoint
    with open(path_to_onnx_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            print("Request ok")
            print(response.json())
        else:
            raise Exception(
                f"Model stealing submit failed. Code: {response.status_code}, content: {response.json()}"
            )


def model_stealing_reset():
    endpoint = f"/modelstealing/reset"
    url = SERVER_URL + endpoint
    response = requests.post(url, headers={"token": TEAM_TOKEN})
    if response.status_code == 200:
        print("Request ok")
        print(response.json())
    else:
        raise Exception(
            f"Model stealing reset failed. Code: {response.status_code}, content: {response.json()}"
        )