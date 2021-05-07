{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Colab 구독 최대한 활용하기",
      "provenance": [],
      "collapsed_sections": [],
      "machine_shape": "hm",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/SunghoonSeok/Study/blob/master/test/test06_dm_colab.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SKQ4bH7qMGrA"
      },
      "source": [
        "# Colab 구독 최대한 활용하기\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yr3WPRYDBWmK",
        "outputId": "dd9e8612-81cc-42f6-86ab-cb61c5aa736d"
      },
      "source": [
        "import os\r\n",
        "from google.colab import drive\r\n",
        "drive.mount('/content/drive/')"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive/\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iJavX6ngB8-5"
      },
      "source": [
        "\r\n",
        "import cv2\r\n",
        "import numpy as np\r\n",
        "from matplotlib import pyplot as plt\r\n",
        "\r\n",
        "for i in range(50000):\r\n",
        "    image_path = '/content/drive/My Drive/dirty_mnist_/dirty_mnist_2nd/%05d.png'%i\r\n",
        "    image = cv2.imread(image_path)\r\n",
        "    image2 = np.where((image <= 254) & (image != 0), 0, image)\r\n",
        "    image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)\r\n",
        "    image4 = cv2.medianBlur(src=image3, ksize= 5)\r\n",
        "    cv2.imwrite('/content/drive/My Drive/dirty_mnist_/dirty_mnist_clean/%05d.png'%i, image4)\r\n",
        "\r\n",
        "for i in range(50000,55000):\r\n",
        "    image_path = '/content/drive/My Drive/dirty_mnist_/test_dirty_mnist_2nd/%05d.png'%i\r\n",
        "    image = cv2.imread(image_path)\r\n",
        "    image2 = np.where((image <= 254) & (image != 0), 0, image)\r\n",
        "    image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)\r\n",
        "    image4 = cv2.medianBlur(src=image3, ksize= 5)\r\n",
        "    cv2.imwrite('/content/drive/My Drive/dirty_mnist_/test_dirty_mnist_clean/%05d.png'%i, image4)\r\n",
        "\r\n"
      ],
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lvT84qUAEwTW"
      },
      "source": [
        "import os\r\n",
        "from typing import Tuple, Sequence, Callable\r\n",
        "import csv\r\n",
        "import cv2\r\n",
        "import numpy as np\r\n",
        "import pandas as pd\r\n",
        "from PIL import Image\r\n",
        "import torch\r\n",
        "import torch.optim as optim\r\n",
        "from torch import nn, Tensor\r\n",
        "from torch.utils.data import Dataset, DataLoader\r\n",
        "from torchinfo import summary\r\n",
        "\r\n",
        "from torchvision import transforms\r\n",
        "from torchvision.models import resnet50\r\n",
        "\r\n",
        "class MnistDataset(Dataset):\r\n",
        "    def __init__(self,dir: os.PathLike,image_ids: os.PathLike,transforms: Sequence[Callable]) -> None:\r\n",
        "        self.dir = dir\r\n",
        "        self.transforms = transforms\r\n",
        "\r\n",
        "        self.labels = {}\r\n",
        "        with open(image_ids, 'r') as f:\r\n",
        "            reader = csv.reader(f)\r\n",
        "            next(reader)\r\n",
        "            for row in reader:\r\n",
        "                self.labels[int(row[0])] = list(map(int, row[1:]))\r\n",
        "\r\n",
        "        self.image_ids = list(self.labels.keys())\r\n",
        "\r\n",
        "    def __len__(self) -> int:\r\n",
        "        return len(self.image_ids)\r\n",
        "\r\n",
        "    def __getitem__(self, index: int) -> Tuple[Tensor]:\r\n",
        "        image_id = self.image_ids[index]\r\n",
        "        image = Image.open(\r\n",
        "            os.path.join(\r\n",
        "                self.dir, f'{str(image_id).zfill(5)}.png')).resize((128,128)).convert('RGB')\r\n",
        "        target = np.array(self.labels.get(image_id)).astype(np.float32)\r\n",
        "\r\n",
        "        if self.transforms is not None:\r\n",
        "            image = self.transforms(image)\r\n",
        "\r\n",
        "        return image, target\r\n",
        "\r\n",
        "transforms_train = transforms.Compose([\r\n",
        "    transforms.RandomHorizontalFlip(p=0.5),\r\n",
        "    transforms.RandomVerticalFlip(p=0.5),\r\n",
        "    transforms.ToTensor(),\r\n",
        "    transforms.Normalize(\r\n",
        "        [0.485, 0.456, 0.406],\r\n",
        "        [0.229, 0.224, 0.225]\r\n",
        "    )\r\n",
        "])\r\n",
        "\r\n",
        "transforms_test = transforms.Compose([\r\n",
        "    transforms.ToTensor(),\r\n",
        "    transforms.Normalize(\r\n",
        "        [0.485, 0.456, 0.406],\r\n",
        "        [0.229, 0.224, 0.225]\r\n",
        "    )\r\n",
        "])\r\n",
        "\r\n",
        "trainset = MnistDataset('/content/drive/My Drive/dirty_mnist_/dirty_mnist_clean/', '/content/drive/My Drive/dirty_mnist_/dirty_mnist_2nd_answer.csv', transforms_train)\r\n",
        "testset = MnistDataset('/content/drive/My Drive/dirty_mnist_/test_dirty_mnist_clean/', '/content/drive/My Drive/dirty_mnist_/sample_submission.csv', transforms_test)\r\n",
        "\r\n",
        "train_loader = DataLoader(trainset, batch_size=128, num_workers=8)\r\n",
        "test_loader = DataLoader(testset, batch_size=32, num_workers=4)\r\n",
        "\r\n",
        "class MnistModel(nn.Module):\r\n",
        "    def __init__(self) -> None:\r\n",
        "        super().__init__()\r\n",
        "        self.resnet = resnet50(pretrained=True)\r\n",
        "        self.classifier = nn.Linear(1000, 26)\r\n",
        "\r\n",
        "    def forward(self, x):\r\n",
        "        x = self.resnet(x)\r\n",
        "        x = self.classifier(x)\r\n",
        "\r\n",
        "        return x\r\n",
        "\r\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\r\n",
        "model = MnistModel().to(device)\r\n",
        "print(summary(model, input_size=(1, 3, 128, 128), verbose=1))\r\n",
        "\r\n",
        "optimizer = optim.Adam(model.parameters(), lr=1e-3)\r\n",
        "criterion = nn.MultiLabelSoftMarginLoss()\r\n",
        "\r\n",
        "num_epochs = 20\r\n",
        "model.train()\r\n",
        "\r\n",
        "for epoch in range(num_epochs):\r\n",
        "    for i, (images, targets) in enumerate(train_loader):\r\n",
        "        optimizer.zero_grad()\r\n",
        "\r\n",
        "        images = images.to(device)\r\n",
        "        targets = targets.to(device)\r\n",
        "\r\n",
        "        outputs = model(images)\r\n",
        "        loss = criterion(outputs, targets)\r\n",
        "\r\n",
        "        loss.backward()\r\n",
        "        optimizer.step()\r\n",
        "\r\n",
        "        if (i+1) % 10 == 0:\r\n",
        "            outputs = outputs > 0.5\r\n",
        "            acc = (outputs == targets).float().mean()\r\n",
        "            print(f'{epoch}: {loss.item():.5f}, {acc.item():.5f}')\r\n",
        "\r\n",
        "submit = pd.read_csv('c:/data/test/dirty_mnist/sample_submission.csv')\r\n",
        "\r\n",
        "model.eval()\r\n",
        "batch_size = test_loader.batch_size\r\n",
        "batch_index = 0\r\n",
        "for i, (images, targets) in enumerate(test_loader):\r\n",
        "    images = images.to(device)\r\n",
        "    targets = targets.to(device)\r\n",
        "    outputs = model(images)\r\n",
        "    outputs = outputs > 0.5\r\n",
        "    batch_index = i * batch_size\r\n",
        "    submit.iloc[batch_index:batch_index+batch_size, 1:] = \\\r\n",
        "        outputs.long().squeeze(0).detach().cpu().numpy()\r\n",
        "    \r\n",
        "submit.to_csv('c:/data/test/dirty_mnist/submission_torch1.csv', index=False)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "23TOba33L4qf",
        "outputId": "d774e593-7178-4252-fc7a-e300a7a129b3"
      },
      "source": [
        "gpu_info = !nvidia-smi\n",
        "gpu_info = '\\n'.join(gpu_info)\n",
        "if gpu_info.find('failed') >= 0:\n",
        "  print('Select the Runtime > \"Change runtime type\" menu to enable a GPU accelerator, ')\n",
        "  print('and then re-execute this cell.')\n",
        "else:\n",
        "  print(gpu_info)"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Fri Feb 12 02:37:20 2021       \n",
            "+-----------------------------------------------------------------------------+\n",
            "| NVIDIA-SMI 460.39       Driver Version: 460.32.03    CUDA Version: 11.2     |\n",
            "|-------------------------------+----------------------+----------------------+\n",
            "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |\n",
            "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |\n",
            "|                               |                      |               MIG M. |\n",
            "|===============================+======================+======================|\n",
            "|   0  Tesla V100-SXM2...  Off  | 00000000:00:04.0 Off |                    0 |\n",
            "| N/A   35C    P0    23W / 300W |      0MiB / 16160MiB |      0%      Default |\n",
            "|                               |                      |                  N/A |\n",
            "+-------------------------------+----------------------+----------------------+\n",
            "                                                                               \n",
            "+-----------------------------------------------------------------------------+\n",
            "| Processes:                                                                  |\n",
            "|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |\n",
            "|        ID   ID                                                   Usage      |\n",
            "|=============================================================================|\n",
            "|  No running processes found                                                 |\n",
            "+-----------------------------------------------------------------------------+\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Sa-IrJS1aRVJ"
      },
      "source": [
        "메모장에서 GPU를 사용하려면 런타임 &#62; 런타임 유형 변경 메뉴를 선택한 다음 하드웨어 가속기 드롭다운을 GPU로 설정하세요."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "65MSuHKqNeBZ"
      },
      "source": [
        "## 추가 메모리\n",
        "\n",
        "<p>Colab Pro를 구독하면 사용 가능한 경우 고용량 메모리 VM에 액세스할 수 있습니다. 고용량 메모리 런타임을 사용하도록 메모장 환경설정을 지정하려면 런타임 &#62; '런타임 유형 변경' 메뉴를 선택한 다음 런타임 구성 드롭다운에서 고용량 RAM을 선택하세요.</p>\n",
        "<p>다음 코드를 실행하여 언제든지 사용 가능한 메모리 용량을 확인할 수 있습니다.</p>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "V1G82GuO-tez"
      },
      "source": [
        "from psutil import virtual_memory\n",
        "ram_gb = virtual_memory().total / 1e9\n",
        "print('Your runtime has {:.1f} gigabytes of available RAM\\n'.format(ram_gb))\n",
        "\n",
        "if ram_gb < 20:\n",
        "  print('To enable a high-RAM runtime, select the Runtime > \"Change runtime type\"')\n",
        "  print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')\n",
        "  print('re-execute this cell.')\n",
        "else:\n",
        "  print('You are using a high-RAM runtime!')"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}