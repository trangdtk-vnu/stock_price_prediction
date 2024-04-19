{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPe90qGJZ0M7AMSyGY6AZqR"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fWmzn2-EVEsp"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "\n",
        "class EnhancedSignAgreementLoss(nn.Module):\n",
        "    def __init__(self, loss_penalty, gain_reward):\n",
        "        super(EnhancedSignAgreementLoss, self).__init__()\n",
        "        self.loss_penalty = loss_penalty\n",
        "        self.gain_reward = gain_reward\n",
        "\n",
        "    def forward(self, y_true, y_pred):\n",
        "        same_sign = torch.eq(torch.sign(y_true), torch.sign(y_pred))\n",
        "        pred_zero = torch.eq(y_pred, 0.0)\n",
        "        actual_pos = torch.gt(y_true, 0.0)\n",
        "        actual_neg = torch.lt(y_true, 0.0)\n",
        "        actual_zero = torch.eq(y_true, 0.0)\n",
        "        condition = torch.where(pred_zero,\n",
        "                                torch.where(actual_zero, torch.tensor(True, device=y_true.device),\n",
        "                                            torch.logical_or(actual_pos, actual_neg)),\n",
        "                                same_sign)\n",
        "        residual = y_true - y_pred\n",
        "        loss = torch.where(condition,\n",
        "                           self.gain_reward * torch.square(residual),\n",
        "                           self.loss_penalty * torch.square(residual))\n",
        "        return torch.mean(loss)"
      ]
    }
  ]
}