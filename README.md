# LFNet_TEST

## LFNet: A Novel Bidirectional Recurrent Convolutional Neural Network for Light-field Image Super-resolution
### Yunlong Wang, Fei Liu, Kunbo Zhang, Guangqi Hou, Zhenan Sun, Tieniu Tan

@ARTICLE{

8356655, 

author={Y. Wang and F. Liu and K. Zhang and G. Hou and Z. Sun and T. Tan}, 

journal={IEEE Transactions on Image Processing}, 

title={LFNet: A Novel Bidirectional Recurrent Convolutional Neural Network for Light-Field Image Super-Resolution}, 

year={2018}, 

volume={27}, 

number={9}, 

pages={4274-4286}, 

doi={10.1109/TIP.2018.2834819}, 

ISSN={1057-7149}, 

month={Sept},}

[webpage](https://ieeexplore.ieee.org/document/8356655/)



## Datasets
The dataset `6scene_tip` for quantitative evaluations (see `TABLE II` in the paper) can be downloaded from [here](https://pan.baidu.com/s/1i5ZnXRn).
It contains 6 LF scenes (`Stone`,`Bush`,`Glass`,`Door`,`Pillar`,`Flower`) which are stored in `.mat` files.
Ground truth 4D LF data will be loaded into `gt_data` variable, while its couterpart LR data will be loaded into `lr_data`.

## Dependencies
- [x] [theano](http://www.deeplearning.net/software/theano/)
- [x] [scikit-image](http://scikit-image.org/)
- [x] [numpy](http://www.numpy.org/)
- [x] [h5py](http://www.h5py.org/)
- [x] [argparse](https://docs.python.org/3/library/argparse.html)

Of course, you can install these packages with `pip install` command

```
sudo pip install argparse h5py numpy scikit-image theano
```

## Usage
Run the following command in terminal to evaluate the pre-trained LFNet model on `6scenes_tip`

```
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 python LFNet_Test_Mat_With_log.py --path ./6scenes_tip --scene Stone Glass Pillar Flower Bush Door --model_path ./model -F 4 -T 7 -C 7 -S
```
* `THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32` specifies configurations of [Theano packages](http://www.deeplearning.net/software/theano/)
* `--path` will load the datasets for evalution from this path
* `--scene` stands for the namelist of LF scenes
* `--model_path` will load the pre-trained models for evaluation from this path
* `-F` stands for upsampling factor (default 4 as in the paper)
* `-T` specifies angular resolution of training LF data (only support choices from [7,9])
* `-C` specifies angular resolution of LF data for evaluation
* `-S` save results

The results will be saved under the folder named `6scenes_tip_eval_l7_f4` in this script.
Meanwhile, a `.log` file named `LFNet_Test.log` and a `.mat` file named `performance_stat.mat` will be generated as output, recording details of the evaluation process (`date`, `model options`, `PSNR`, `SSIM`, `Elapsed Time` and so on)





