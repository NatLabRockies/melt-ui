import numpy as np
import torch
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from ptmelt.utils.preprocessing import get_normalizers
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

router = APIRouter()


@router.post("/normalize_data")
async def normalize_data(request: Request):
    body = await request.json()

    input_data = np.asarray(body.get("input_data"))
    normalizer_type = body.get("norm_typ")

    normalizer_list = get_normalizers(norm_type=normalizer_type, n_normalizers=1)
    normalizer = normalizer_list[0]
    normalizer.fit(input_data)

    scaled_data = normalizer.transform(input_data)

    return JSONResponse(
        content={
            "scaled_data": scaled_data.tolist(),
            # "scaler_object": scaler_object,
        }
    )


# @router.post("get_supervised_dataloaders")
# async def get_supervised_dataloaders(request: Request):
#     body = await request.json()

#     # unpack settings from node
#     x = np.asarray(body.get("x"))
#     y = np.asarray(body.get("y"))
#     val_size = float(body.get("val_size", 0.1))
#     test_size = float(body.get("test_size", 0.1))
#     random_state = int(body.get("random_state", 42))
#     # TODO: scaler to precompute un-normalized values
#     shuffle = bool(body.get("shuffle", True))
#     batch_size = int(body.get("batch_size", 32))

#     # compute first test size as sum of val and test
#     test_size_combined = val_size + test_size

#     # do the splits first
#     x_train, x_tmp, y_train, y_tmp = train_test_split(
#         x, y, test_size=test_size_combined, random_state=random_state
#     )

#     # then split the temp into val and test
#     relative_test_size = test_size / test_size_combined
#     x_val, x_test, y_val, y_test = train_test_split(
#         x_tmp, y_tmp, test_size=relative_test_size, random_state=random_state
#     )

#     # TODO: precompute the un-normalized values if scaler is provided

#     # Form the dataloaders for supervised learning
#     train_dataset = TensorDataset(
#         torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
#     )
#     val_dataset = TensorDataset(
#         torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float()
#     )
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     return JSONResponse(
#         content={
#             "train_
