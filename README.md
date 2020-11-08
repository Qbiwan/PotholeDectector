# Pothole Detector

## Directory

```
Pothole_Detector
├── model                                         
│     └── model.bin                               <- trained model  
├── Dataset                                       <- Kaggle Dataset
│     ├── normal                                  <- normal images
│     │     └── *.jpg      
│     └── potholes                                <- pothole images
│           └── *.jpg 
├── Dataset224                                    <- resized images to (224,224)
│     ├── normal                                  
│     │     └── *.jpg      
│     ├── potholes                                
│     │     └── *.jpg 
│     └── images_labeled.csv                      <- csv with labeled imagepaths
├── src
│     ├── static                                    
│     │      ├── css                              <- bootstrap css files 
│     │      │    ├── *.css
│     │      │    └── *.css.map 
│     │      ├── js                               <- bootstrap js files  
│     │      │    ├── *.js
│     │      │    └── *.js.map 
│     │      └── images_folder
│     ├── templates                               <- flask html file
│     │      └── index.html
│     ├── app.py                                  <- flask app
│     ├── main.py                                 <- training    
│     └── resize.py                               <- resizing image
└── README.md
```
## Data

* The [`Kaggle dataset`](https://www.kaggle.com/atulyakumar98/pothole-detection-dataset) contains two folders - normal and potholes. 'normal' contains images of smooth roads from different angles and 'potholes' contains images of roads with potholes in them. This dataset is stored in `Pothole_Detector/Dataset`

* The above dataset is resized using [resize.py](src/resize.py) to (224,224) and stored in `Dataset224/`
* [images_labeled.csv](Dataset224/images_labeled.csv) is a pandas dataframe created from `resize.py` that has two columns - image paths and labels


## Getting Started

### Installation

Create the virtual environent using conda. 

```bash
$ conda env create -f environment.yml
$ conda activate PotholeDetector

# then run the python scripts
$ python resize.py # resize images to 224
$ python main.py   # train model
$ ptthon app.py    # serve flask app
```



