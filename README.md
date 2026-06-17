# Coffee Machine Energy Disaggregation & Prediction

This repository contains a Machine Learning and Deep Learning project focused on energy disaggregation (Non-Intrusive Load Monitoring - NILM) for a coffee machine using TensorFlow and Keras.

The main objective is to isolate and predict the individual power consumption of the coffee machine from the total aggregated household electricity signal.

## 👥 The Team (Contributors)
* **Georgios Karanikolopoulos** ([@giorgoskaranikolopoulos](https://github.com/giorgoskaranikolopoulos)) 
* **Nikolaos Karageorgis** ([@karagionikos](https://github.com/karagionikos)) 
* **Nikolaos Karanikas** ([@karamessi10](https://github.com/karamessi10)) 

---

## 🚀 Code Pipeline & Features

The implementation is built using Python inside a Google Colab environment (with GPU acceleration) and covers the following pipeline stages:

1. The project expects the following dataset files (provided in `.txt` format):
* `Input_Data.txt`: Aggregated household power consumption (features).
* `Output_Data.txt`: True power consumption of the coffee machine (targets).
* `CoffeeMachinemaxAgg.txt`: Maximum aggregated power value (used for normalization/denormalization).
* `CoffeeMachinemaxApp.txt`: Maximum appliance power value (used for normalization/denormalization).
2. **Data Normalization:** Scales the raw dataset arrays based on their respective maximum values to achieve stable and faster neural network convergence.
3. **Train/Test Dataset Split:** * **80%** dedicated for training (Train set: 40,000 samples).
   * **20%** reserved for model evaluation (Test set: 10,000 samples).
4. **3D Tensor Reshaping:** Automatically reshapes the data vectors into 3D Tensors, making them compatible with Recurrent Neural Network layers (LSTM / GRU) or Convolutional layers (1D CNN) in Keras.
5. **Data Visualization:** Employs an intelligent thresholding technique to find activation windows where the coffee machine is actively running. Generates clean dual y-axis plots contrasting total household load against the isolated appliance load.

---

## 🛠️ Requirements & Installation

To run this Jupyter Notebook locally or in your own environment, make sure you have the following packages installed:

```bash
pip install numpy matplotlib tensorflow
