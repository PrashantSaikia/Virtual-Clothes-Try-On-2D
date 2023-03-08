# Virtual-Clothes-Try-On-2D

If you're running it on Mac OS with M1 or M2 chip, run `app_mac.py`; the GPU accelarator is `mps` in Mac as opposed to `CUDA` otherwise. And also, `pipe.enable_attention_slicing()` is added to make it run a little faster. Takes around 10x longer on M1 than running on CUDA. 

P.S. If you get error when downloading large model files in Google Colab, install `gdown` the following way and try again:

`pip install --upgrade --no-cache-dir gdown`

Or simply calling  `gdown "<drive-id>&confirm=t"` might also work.

## Demo

![image](https://user-images.githubusercontent.com/39755678/223530196-d5d5f45d-9180-497a-9f8e-c5e00e3060d7.png)

![image](https://user-images.githubusercontent.com/39755678/223536038-fe8dc624-7729-41e4-bb61-5340720a3c0c.png)

## Usage:
```
git clone https://github.com/PrashantSaikia/Virtual-Clothes-Try-On-2D.git
cd Virtual-Clothes-Try-On-2D
pip install -r requirements.txt
python app.py 
#python app_mac if running on Mac OS with M1 or M2 chip
```
