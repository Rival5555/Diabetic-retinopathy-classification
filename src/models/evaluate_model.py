import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd

def shap_explain(model, dataloader):
    model.eval()
    batch = next(iter(dataloader))[0][:10].to(device)
    explainer = shap.GradientExplainer(model, batch)
    shap_values = explainer.shap_values(batch)
    shap.image_plot(shap_values, batch.cpu().numpy())
