import click
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import FolderDatasetAdaptiveAug
from model import Classifier
import pytorch_lightning as pl



class PredictClassifier(Classifier):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat


@click.command()
@click.option("--ckpt_path", default=None, required=True)
@click.option("--save_to_onnx", default=False)
def main(ckpt_path: str, save_to_onnx: bool):
    model = PredictClassifier.load_from_checkpoint(ckpt_path)
    if save_to_onnx:
        torch_input = torch.randn(1, 3, 256, 256)
        onnx_program = torch.onnx.dynamo_export(model, torch_input)
        onnx_program.save("my_image_classifier.onnx")
        print("onnx program saved!")
        return

    trainer = pl.Trainer(max_epochs=1, devices=[0])
    dataset = FolderDatasetAdaptiveAug(
        "./data"
    )
    ans = trainer.predict(model, DataLoader(dataset, batch_size=256, shuffle=False, num_workers=127))
    ans = torch.cat(ans)
    ans = ans.numpy()
    print(f"Number of predicted vectors: {len(ans)}")
    print(f"Example of vector: {ans[0]}")
    for i in range(ans.shape[1]):
        print(f"quants for {i}:")
        quantize_to_int8(ans[:, i])


class Quant:
    def __init__(self, upper, lower, value):
        self.upper = upper
        self.lower = lower
        self.value = value

    def __str__(self):
        return str(float(self.value))


def quantize_to_int8(nums: np.ndarray):
    quants = []
    nums.sort()
    # qsize = len(nums)//255
    qsize = (nums[-1] - nums[0]) / 255
    error = 0

    for i in range(254):
    #     upper = nums[max(qsize * (i + 1) - 1, len(nums)-1)]
    #     lower = nums[qsize * i]
    #     value = (upper + lower) / 2
        q = nums[nums >= nums[0] + qsize * i]
        q = q[q <= nums[0] + qsize * (i + 1)]
        if not len(q):
            continue
        upper = q[-1]
        lower = q[0]
        value = (upper + lower) / 2
        for j in q:
            error += abs(value - j)
        quants.append(Quant(upper, lower, value))
    print(f"Error: {error / len(nums)}")
    for i in range(10):
        print(str(quants[i]))



if __name__ == "__main__":
    main()
