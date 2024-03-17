import torch
import pandas as pd


def save_model(model, hour):
    filename = f"./models/model_{hour}"
    # SAVE TO PTH
    torch.save(model.state_dict(), filename + ".pth")

    # SAVE TO ONNX
    torch.onnx.export(
        model.to(torch.device("cpu")),
        torch.randn(1, 3, 32, 32),
        filename + ".onnx",
        export_params=True,
        input_names=["x"],
    )


def save_history(history, loss_name, hour):
    filename = f"./reports/report_{hour}"
    df = pd.DataFrame(history)
    df.columns = [loss_name, "l2_loss"]
    df.to_csv(filename + ".csv", index=False)


def get_position_by_id(ids, dataset):
    df = pd.DataFrame([dataset.ids]).T
    df.columns = ['id']
    return torch.from_numpy(df.index[df['id'].isin(ids)].values)
