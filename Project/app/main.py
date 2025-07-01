from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from SquareNet import SquareNet

model = SquareNet()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
app = FastAPI()

class Input(BaseModel):
    x: float


@app.post("/predict")
def predict(data: Input):
    tensor_in = torch.tensor([[data.x]], dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor_in)
    return JSONResponse(status_code=200, content={'predicted': round(output.item())})
