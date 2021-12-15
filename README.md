# AirPose
## AirPose: Multi-View Fusion Network for Aerial 3D Human Pose and Shape Estimation

![iccv_teaser](https://user-images.githubusercontent.com/19806758/145577115-c1f08e0b-527e-4ada-bbbf-2c5d0dde632b.png)
_________

Check the teaser [video](https://www.youtube.com/watch?v=gUKMepNm-HQ/)

This repository contains the code of _AirPose_ a novel markerless 3D human motion capture (MoCap) system for unstructured, outdoor environments that uses a team of autonomous unmanned aerialvehicles (UAVs) with on-board RGB cameras and computation.

Please clone the repository with the following 

`git clone https://github.com/robot-perception-group/AirPose.git --recursive`

_________

Data can be freely accessed [here](https://keeper.mpdl.mpg.de/d/1cae0814c4474f5a8e19/).
Please download the data, and untar it.

_________

To run the code of this repository you first need to download the data.
The code was tested using `Python 3.8`.

`SMPLX` code in this repo is a modified version of the official SMLX [implementation](https://github.com/vchoutas/smplx). Download the SMPLX model weights from [here](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip) and put them in `copenet/src/copenet/data/smplx/models/smplx`. You need to register before being able to download the weights.

Now, you may want to create a virtual environment. Please be sure your `pip` is updated.

Install the necessary requirements with `pip install -r requirements.txt`. If you don't have a cuda compatible device, change the device to `cpu` in `copenet_real/src/copenet_real/config.py` and `copenet/src/copenet/config.py`.

Download the Head and hands indices files form [here](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_mano_flame_correspondences.zip) and place them in `copenet/data`.

## Synthetic data training 

The data to be used is `copenet_synthetic.tar.gz`

`cd AirPose/copenet/`

`pip install -e .`

And code can be run by the following

`python src/copenet/copenet_trainer.py --name=test_name --version=test_version --model=muhmr --datapath=path/location --log_dir=path/location/ --copenet_home= absolute path to the copenet directory --optional-params...`

The `datapath` is the location of the training data.

`--model` specify the model type between `[hmr, muhmr, copnet_singleview, copenet_twoview]` which corresponds to the Baseline, Baseline+multi-view, Baseline+Fullcam and AirPose respectively.

Logs will be saved in `$log_dir/$name/$version/`

`optional-params` is to be substituted with the `copenet_trainer` available params as weights, lr..

## Evaluation on the synthetic data
```
cd AirPose/copenet_real/src/copenet_real

python src/copenet_real/scripts/copenet_synth_res_compile.py "model type" "checkpoint Path" "/path to the dataset"
```
For model type `[muhmr, copenet_twoview]`.

For model type `[hmr, copenet_singleview]`, the provided checkpoint is trained with an older pytorch lightning version (<1.2). If you want to use them, install pytorch-lightning<=1.2. We provide the precalculated outputs on the syntehtic data using these checkpoints. To generate the metrics, run
```
python src/copenet_real/scripts/copenet_synth_res_compile.py "model type" "checkpoint directory path" "/path to the dataset"
``` 


## Fine-tuning on real dataset
The data to be used is `copenet_dji.tar.gz`.

`cd AirPose/copenet_real/`

`pip install -e .`

Install the human body prior from [here](https://github.com/nghorbani/human_body_prior) and download its pretrained weights (version 2) from [here](https://smpl-x.is.tue.mpg.de/download.php). Set the `vposer_weights` variable in the `copenet_real/src/copenet_real/config.py` file to the absolute path of the downloaded weights. 

And code can be run by the following

`python src/copenet_real/copenet_trainer.py --name=test_name --version=test_version --model=muhmr --datapath=path/location --log_dir=path/location/ --resume_from_checkpoint=path to the pretrained checkpoint --copenet_home= absolute path to the copenet directory --optional-params...`

The `datapath` is the location of the training data.

`--model` specify the model type between `[hmr, copenet_twoview]` which corresponds to the Baseline, AirPose respectively.

*Note*: for the `hmr` model `pytorch-lightning<=1.2` is required.

The `--resume_from_checkpoint` is path to the pretrained checkpoint on the synthetic data. 


## Testing the client-server synchronization mechanism

To this end you need to install ros-{melodic,noetic} in your pc.

Please follow the instructions that you can find [here](http://wiki.ros.org/Installation/Ubuntu)

After that you need to install the following dependencies:

```
sudo add-apt-repository ppa:joseluisblancoc/mrpt-stable
sudo apt install libmrpt-dev mrpt-apps
```

Navigate to your `catkin_ws` folder (e.g. `AirPose/catkin_ws`) and run:

```
touch src/aircap/packages/optional/basler_image_capture/Grab/CATKIN_IGNORE
touch src/aircap/packages/optional/ptgrey_image_capture/Grab/CATKIN_IGNORE
```

### this applies to ros-melodic
```
cd catkin_ws/src/aircap
git checkout realworld-airpose-melodic-backport
cd ../../
touch src/aircap/packages/3rdparty/mrpt_bridge/CATKIN_IGNORE
touch src/aircap/packages/3rdparty/pose_cov_ops/CATKIN_IGNORE
sudo apt install -y ros-melodic-octomap-msgs ros-melodic-cv-camera ros-melodic-marker-msgs ros-melodic-mrpt-msgs ros-melodic-octomap-ros ros-melodic-mrpt-bridge
```

### this applies to ros-noetic
`sudo apt install -y ros-noetic-octomap-msgs ros-noetic-cv-camera ros-noetic-marker-msgs ros-noetic-mrpt-msgs ros-noetic-octomap-ros`

Then you can run `catkin_make` from the `catkin_ws` folder to build the whole workspace.

To run the client-server architecture you need:
- An image topic
- A camera_info topic
- A feedback topic with the region of interest information

To test the code you can do the following.
- Download the dji rosbags that you can find [here]().
- In separated terminals (with the workspace sourced) run:
  - `roscore`
  - `rosparam set use_sim_time true`
  - `roslaunch client1`
  - `roslaunch client2`
  - `python server1`
  - `python server2`
  - `python visualization`
  - `rosbag play d*_split1.bag --clock --pause`, where split{id} is the n-th split of a longer sequence. The splits have some overlap between them.

At this point you should be able to see play the rosbag in the way you prefer.

The published topics, for each machine, are:
- `machine_x/step1_pub`, the results of the first step of the network, read by the other machine
- `machine_x/step2_pub`, the results of the second step of the network, read by the other machine
- `machine_x/step3_pub`, the final results of `machine_x`
