# 🐶🐱 Cat-Dog Classification 
AI course project. based on VGG16 Transfer Learning





## For Group Teammates  *(I will delete this part next time update) 25-3-22*

Hey guys, I'm happy to form a group with you. 😃

Before you start reading the project-related materials, I just want to let you know that, in my opinion, the priorities for our team cooperation are:

1. Everyone feels happy during the teamwork
2. Everyone can learn something
3. We can get a good grade from the professor

So, if you have any ideas, complaints, suggestions, questions, or doubts, feel free to talk. Let me know what you think.

I don’t think I’m the team leader. Everyone is a team leader. I just happened to be the one with the idea.

By the way, if you don’t have enough time or energy for the project, I totally understand. Living in Vancouver is difficult. Just let me know — I can help with your part. This project is simple, and the content is not too complex. I’m interested in this field, so I can do more if needed.

If you want to learn something but don’t have related background, don’t worry — me too. I just started learning Python this semester. Feel free to ask questions in the group. We can learn more during discussions.

Thank you for your kindness and cooperation in advance!🤗



## What Our Team Could Do Next  *(just a rough idea — I haven’t planned them carefully.) 25-3-22*

1. Review and modify the code to make it simpler and easier to understand.   (Send me your GitHub username and I’ll add you to this project so you can edit it directly)
2. Improve the interface to make it look better.  
3. In our presentation slides, explain the code (especially `train.py`).  
4. In our presentation slides, explain the following concepts: CNN, transfer learning, VGG16, TensorFlow, overfitting, supervised learning, binary cross-entropy loss, etc.  
   (To be honest, I don’t fully understand these concepts either. ChatGPT just told me that I should know them... Here's the full list it gave me.)

### ***📌 Machine Learning***

- **Supervised Learning**
- **Binary Cross-Entropy Loss**
- **Overfitting** and how to prevent it (e.g., **Dropout**, **Data Augmentation**)
- **Model evaluation metrics**: Accuracy, Precision, Recall, F1-score

### ***📌 Computer Vision***

-  **Basic CNN structure** (Convolutional layers, Pooling layers, Fully connected layers)
- **Transfer Learning** (using **VGG16**)
-  **Image Preprocessing** (Data Augmentation, Normalization, Noise Handling)

### ***📌 Python & Deep Learning***

-  `TensorFlow / Keras`
-  Image data handling with `NumPy`, `Pandas`, `OpenCV`
-  Visualizing training results with `Matplotlib`





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





## Performance Showcase

![](catdog.gif)



