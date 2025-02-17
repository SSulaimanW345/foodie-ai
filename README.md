<div align="center">

  <h3 align="center">A Food Image Classification Project</h3>
</div>

## <a name="introduction">🤖 Overview </a>

This project leverages TensorFlow and EfficientNet to build a powerful food image classification model using the FOOD101 dataset. It applies advanced food image processing techniques, including data preprocessing, mixed training, and feature extraction, to enhance model accuracy and performance. The model's training process is visualized with TensorBoard for detailed insights into its learning progress.
## <a name="features">🔋 What have we covered</a>

- 📦 TensorFlow library
- 📚 TensorFlow dataset: FOOD101 downloaded and explored from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/overview)
- 🔧 Creating preprocessing functions to prepare data to be used by our model.
- ⚙️ Developing a TensorFlow data input pipeline for optimizing performance (prefetching).
- 🧪 Batching & preparing datasets for modeling.
- 🔄 Creating modeling callbacks (i.e., TensorBoard, ModelCheckpoint, EarlyStopping).
- ⚡ Making use of a new TensorFlow feature called MIXED TRAINING for efficient training.
- 🏋️ Using a Pre-trained model EfficientNet (a state-of-the-art computer vision architecture from 2019).
- 🧠 Building and training a Feature extraction model for our classification problem.
- 🔥 Fine-tuning the feature extraction model developed.
- 📊 Using TENSORBOARD to visualize our model(s) training results.

and many more, including code architecture and reusability 

## <a name="quick-start">🤸 Quick Start</a>

Follow these steps to set up the project locally on your machine.

**Prerequisites**

Make sure you have the following installed on your machine:

- [Git](https://git-scm.com/)
- [Python](https://www.python.org/downloads/)
- [Jupyter Notebook](https://jupyter.org/install) 

**Cloning the Repository**

```bash
git clone https://github.com/SSulaimanW345/foodie-ai.git

```

**Installation**

Install the project dependencies using pip:

```bash
pip install -r requirements.txt
pip install tensorflow pandas numpy matplotlib

```
**Opening and Running the Jupyter Notebook**
```bash
jupyter notebook foodie-ai.ipynb
```


**Running the Project**

```bash
npm run dev
```
