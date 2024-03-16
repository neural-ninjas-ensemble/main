import torch
import pandas as pd
from datetime import datetime


def save_model(model):
    now = datetime.now()
    filename = f"./models/model_{now.hour}:{now.minute}"

    # SAVE TO PTH
    torch.save(model.state_dict(), filename + ".pth")

    # SAVE TO ONNX
    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        filename + ".onnx",
        export_params=True,
        input_names=["x"],
    )


def save_history(history, loss_name):
    now = datetime.now()
    filename = f"./reports/{loss_name}_{now.hour}:{now.minute}"

    df = pd.DataFrame(history)
    df.columns = [loss_name, "l2_loss"]
    df.to_csv(filename + ".csv", index=False)
