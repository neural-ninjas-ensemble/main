import torch
import pandas as pd
from datetime import datetime


def save_model(model):
    now = datetime.now()
    filename = f"./models/model_{now.hour}:{now.minute}"

    model.eval()
    # SAVE TO PTH
    torch.save(model.state_dict(), filename + ".pth")

    # SAVE TO ONNX
    torch_input = torch.randn(1, 3, 32, 32)
    onnx_program = torch.onnx.dynamo_export(model, torch_input)
    onnx_program.save(filename + ".onnx")


def save_history(history):
    now = datetime.now()
    filename = f"./reports/report_{now.hour}:{now.minute}"

    df = pd.DataFrame(history)
    df.columns = ["cont_loss", "l2_loss"]
    df.to_csv(filename + ".csv", index=False)
