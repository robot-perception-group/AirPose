# AirPose
## AirPose: Multi-View Fusion Network for Aerial 3D Human Pose and Shape Estimation

![iccv_teaser](https://user-images.githubusercontent.com/19806758/145577115-c1f08e0b-527e-4ada-bbbf-2c5d0dde632b.png)
_________

Check the teaser [video](https://www.youtube.com/watch?v=gUKMepNm-HQ/)

This repository contains the code of _AirPose_ a novel markerless 3D human motion capture (MoCap) system for unstructured, outdoor environments that uses a team of autonomous unmanned aerialvehicles (UAVs) with on-board RGB cameras and computation.
_________

Data can be freely accessed [here]()

_________

To run the code of this repository you first need to download the data.
The code was tested using `Python 3.8`.

Please get `smplx` code requesting access [here](https://smpl.is.tue.mpg.de/) and place it in `copenet/src/copenet` and install it `python install -e smplx`.

Install the necessary requirements with `pip install -r requirements.txt`

## Synthetic data training 
`cd copenet`
`ln -s . copenet`
And code can be run by the following
`python copenet_trainer.py --name=test_name --version=test_version --model=muhmr --datapath=path/location --log_dir=path/location/ --optional-params...`

The `datapath` is the location of the training data.
`--model` specify the model type between `[muhmr, copenet_twoview]`
Logs will be saved in `$log_dir/$name/$version/`
