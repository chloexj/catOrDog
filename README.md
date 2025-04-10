# 🐶🐱 Cat-Dog Classification 
Project based on VGG16 Transfer Learning

![](catdog.gif)

## What You Can Start With

There are 2 options:

- (A little bit complex) You want to use this code to train a model by yourself  
- (Super easy) You want to use the model trained by me  
  *(If you don’t have a good GPU or find it hard to set up the environment, this option is easier. The environment setup can be really annoying.)*

---

- ## Option 1: Train the Model Yourself

  1. I use [TensorFlow](https://www.tensorflow.org/install/pip#windows-native), which helps your GPU work during training.  
     If you don’t use a GPU, it will use the CPU by default, and training will take a long time.  
     If your environment is similar to mine (Windows system, NVIDIA GPU), you can just [jump to the Environment section](#Environment).  
     If your environment is different, please check the TensorFlow official website, follow their setup instructions, or ask ChatGPT (or use other tools).  After this step, you should have TensorFlow, NumPy, and your GPU driver installed successfully.

     **Tip**: Carefully check your GPU driver version, TensorFlow version, NumPy version, and Python version before installing.  
     I spent the whole night installing and uninstalling things 😩

  2. Download the [training data from Kaggle (around 1GB)](https://www.kaggle.com/datasets/salader/dogs-vs-cats/data).  
     You can also try other datasets from the same website.  
     Create a folder named `dataset`, after downloading, put the files inside it.

     Follow the structure shown in [Project Structure](#Project-Structure).

  3. Download my code manually, unzip it, and open it with your IDE (I use PyCharm).  
     If you're familiar with Git, you can just run:

     ```bash
     git clone https://github.com/chloexj/catOrDog.git
     ```

  4. Install the required packages:

     ```bash
     pip install keras matplotlib opencv-python scikit-learn
     ```

  5. Run `test.py` to check if TensorFlow is working properly.

  6. Run `train.py` to start training the model.

  7. Run `gui.py` to see the final result.



- ## Option 2: Use the Model Trained by Me

  1. Download my code manually, unzip it, and open it with your IDE (I use PyCharm).  
     If you're familiar with Git, you can just run:

     ```bash
     git clone https://github.com/chloexj/catOrDog.git
     ```

  2. Download the model I trained -> [Model Download Link](https://drive.google.com/file/d/1WL8kfo1W2DGWLFvVodDp8hppsxkKN9r9/view?usp=drive_link), 

     Create a folder named `models`, and put the downloaded file inside it.
     Follow the structure shown in [Project Structure](#Project-Structure).

  3. Install the required packages:

     ```bash 
     pip install tensorflow-cpu numpy opencv-python pillow
     ```

  4. run `gui.py` to check final result. 



## Project Structure

```
CatDogTest/
├── models/  
│   └── vgg16_dog_cat_classifier.h5   # For Option2: If you can’t train, download it from link I provided
│                                     # For Option1: don't need download
│                                     # After running `train.py`, this file will be created automatically.                                  
├── dataset/   # For Option1: download it from link I provided
│   ├── test/
│   │   ├── cats/
│   │   └── dogs/
│   └── train/
│       ├── cats/
│       └── dogs/

├── gui.py         # A simple (and maybe ugly 😅) interface with upload and prediction features.
├── train.py       # Core code using VGG16 to train the model. We'll explain this in the slides.
├── test.py        # For testing if TensorFlow runs properly with your GPU.
├── predict.py     # Useless now, Old test for prediction before GUI was created.
├── visualize.py   # Useless now, Originally used for plotting training history (currently not working).
├── .gitignore
└── README.md
```





## Environment

Windows 10

GPU: notebook NVIDIA RTX2060 (Training takes around 30min)

IDE: pycharm

python 3.9

numpy 1.23.5

tensorflow 2.10

```aiignore
pip install tensorflow==2.10

pip install numpy==1.23.5
```

[**NVIDIA GPU Driver - CUDA Toolkit  11.2**](https://developer.nvidia.com/cuda-11.2.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)

[**NVIDIA GPU Driver - cuDNN 8.1.0**](https://developer.nvidia.com/rdp/cudnn-archive)

[Training Data - kaggle (around 1GB)](https://www.kaggle.com/datasets/salader/dogs-vs-cats/data)

Other packages:
```aiignore
pip install keras matplotlib opencv-python scikit-learn

```









