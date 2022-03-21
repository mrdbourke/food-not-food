# Food Not Food dot app (🍔🚫🍔)

Code for building a machine Learning powered app to decide whether a photo is of food or not.

See it working live at: https://foodnotfood.app

Yes, that's all it does.

It's not perfect.

But think about it.

How do you decide what's food or not?

## Inspiration

Remember hotdog not hotdog?

<img src="images/hotdog-not-hotdog.jpeg"/> 

That's what this repo builds, excepts for food or not.

It's arguably harder to do food or not.

Because there's so many options for what a "food" is versus what "not food" is.

Whereas with hotdog not hotdog, you've only got one option: is it a hotdog or not?

## Video and notes

I built this app during a 10-hour livestream to celebrate 100,000 YouTube Subscribers (thank you thank you thank you). 

The full stream replay is [available to watch on YouTube](https://youtu.be/W5XNOmyJr6I).

The code has changed since the stream.

I made it cleaner and more reproducible.

My notes [are on Notion](https://www.notion.so/mrdbourke/November-6-100k-Livestream-Celebration-a6ed0836c7a9490f94ada8891e606d8e).

## TODO: Steps to reproduce

**Note:** If this doesn't work, please [leave an issue](https://github.com/mrdbourke/food-not-food/issues).

To reproduce, the following steps are best run in order.

You will require and installation of Conda, I'd recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Clone the repo

```
git clone https://github.com/mrdbourke/food-not-food
cd food-not-food
```

### Environment creation

I use Conda for my environments. You could do similar with [`venv`](https://docs.python.org/3/library/venv.html) and `pip` but I prefer Conda.

This code works with Python 3.8.

```
conda create --prefix ./env python=3.8 -y
conda activate ./env
conda install pip
``` 

### Installing requirements




2. Install `requirements.txt`

3. Download data -> "data_download" (warning this takes ~10Gb of download/storage, this can be removed after model training)
4. Process data -> "data_processing"
5. Train model -> "model_building"
6. Model eval -> "model_eval" (test images in "test_food_not_food_images")
7. Web app version (see index.html, script.js, styles.css)


## TODO: Dataset(s)

Want to build a dataset of 10,000~ food and 10,000~ not food images (these numbers can go up if needed).

* Food images = Food101
* Not images = Open Images (random subset + filtering for not food images)

**Food-5k (2500 images of food and 2500 images of not food)**
* Downloaded data from Food not food on Kaggle (food images are prefixed with `1` e.g. `1_340.jpg` and not food images are prefixed with `0`) - https://www.kaggle.com/binhminhs10/food5k 
* Extract files from food5k into `food_images` and `not_food_images` using `extract_food_5k.py` (note: this disregards the original train/eval/test splits of food5k)

**Open Images**
* Installed FiftyOne to download images from Open Images - `pip install fiftyone`
* Following this guide to download a random subset of images - https://voxel51.com/docs/fiftyone/tutorials/open_images.html
* Downloaded ~100 sample images from Open Images using `download_open_images.py` using `seed=42`
* **Next:** Figure out how to sort through Open Images data and then create a dataset of random images from Open Images into `not_food_images`
