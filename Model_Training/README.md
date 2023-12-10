# Model Training

This repository contains the code and resources for training a Behavior Cloning Model for auto-piloting rosbotxl.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this project, we aim to train a Dave-v2 model for rosbotxl to navigate through building. 
The model architecures are inside the model.py file. We have implemented and deployed 8 models. There are mainly four types:
 - **Original Dave2** model with 5 conv layers and ELU activation function.
 - **Dave2v3** with RELE activation, 3 conv layers, and max-pooling between layers.
 - **Dave2resnet** replace original conv encoder by a resnet 18 conv network.
 - **DAVE2lstmModel** added a LSTM layer after the conv layer, and it is based on Dave2resnet.
Among them, the best model is DAVE2resnetModel in our case.

## Installation

To install the necessary dependencies, follow these steps:

1. Clone this repository to your local machine.
2. Install the required libraries and packages by running the following command:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

To train the model, follow these steps:

1. Put your collected datasets in a subfolder \<Your Folder\> under Model_Training folder


2. The dataset folder should have following structure. Ensure your \<run> folder has same name as \<run>.csv
    - /\<Your Folder\>
        - /\<run1>
            - /image1
            - /image2
            ...
        - /\<run2>
            - /image1
            - /image2
            ...
        - \<run1>.csv
        - \<run2>.csv


3. Run the training script under Model_Training folder:

    ```shell
    python train.py \<Your Folder\> 
    ```
    Specific choice of batch size or number of iterations can be changed based on options. Please check train.py file for more information

    - `dataset`: This is a required argument. It specifies the parent directory of the training dataset.
    - `--batch`: This optional argument specifies the batch size for training. The default value is 64.
    - `--epochs`: This optional argument specifies the number of epochs for training. The default value is 100.
    - `--lr`: This optional argument specifies the learning rate for the optimizer. The default value is 0.0005.
    - `--robustification`: This optional argument specifies whether to use robustification during training. Current robustification has 50% change of flip the image and the steering angle, plus 20% chance of adding gaussian noise to the image. The default value is False.
    - `--noisevar`: This optional argument specifies the noise variance for robustification. The default value is 15.
    - `--save_folder`: This optional argument specifies the directory where the trained model will be saved. The default value is "v4".


3. Monitor the training progress and evaluate the model's performance. 


## Inference
After training, you should have many pt checkpoints inside the save_folder you specified. 
1. Copy the checkpoints to the `\ros2_tutorial_workspace\src\python_package_with_a_node\python_package_with_a_node\model_cpt`
folder on your rosbotxl.
2. Change the inference model checkpoint name in your 
`\ros2_tutorial_workspace\src\python_package_with_a_node\python_package_with_a_node\rosbot_ml_node.py` file.

3. follow the instruction to run inference.


## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
