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
Please download the data, and untar it whenever necessary.

_________

The code was tested using `Python 3.8`.

`SMPLX` code in this repo is a modified version of the official SMLX [implementation](https://github.com/vchoutas/smplx). Download the SMPLX model weights from [here](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip) and run the following

```
# from the download location
unzip models_smplx_v1_1.zip -d models_smplx
unzip models_smplx/models/smplx/smplx_npz.zip -d used_models
rm models_smplx -r
```

Then copy the *content* of `used_models` (just created, with `SMPLX_{MALE,FEMALE,NEUTRAL}.npz` files) folder into `your_path/AirPose/copenet/src/copenet/data/smplx/models/smplx`.

You need to register before being able to download the weights.

Now, you may want to create a virtual environment. Please be sure your `pip` is updated.

Install the necessary requirements with `pip install -r requirements.txt`. If you don't have a cuda compatible device, change the device to `cpu` in `copenet_real/src/copenet_real/config.py` and `copenet/src/copenet/config.py`. Check out [this](https://stackoverflow.com/questions/65637222/runtimeerror-subtraction-the-operator-with-a-bool-tensor-is-not-supported) link to fix the runtime error `RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported` due to the `Torchgeometry` package.

Download the Head and hands indices files form [here](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_mano_flame_correspondences.zip) and place them in `copenet/data/smplx` (`MANO_SMPLX_vertex_ids.pkl` and `SMPL-X__FLAME_vertex_ids.npy`).

## Synthetic data training 

The data to be used is `copenet_synthetic.tar.gz`

### Preprocess
To run the code of this repository you first need to preprocess the data using

```
# from AirPose folder
python copenet/src/copenet/scripts/prepare_aerialpeople_dataset.py /absolute/path/copenet_synthetic
```

`cd AirPose/copenet/`

`pip install -e .`

And code can be run by the following

`python src/copenet/copenet_trainer.py --name=test_name --version=test_version --model=muhmr --datapath=/absolute/path/copenet_synthetic --log_dir=path/location/ --copenet_home= absolute path to the copenet directory --optional-params...`

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

Install the human body prior from [here](https://github.com/nghorbani/human_body_prior) and download its pretrained weights (version 2) from [here](https://smpl-x.is.tue.mpg.de/download.php). Set the `vposer_weights` variable in the `copenet_real/src/copenet_real/config.py` file to the absolute path of the downloaded weights. If you do NOT have a GPU please change `human_body_prior/tools/model_loader.py` line 68 from `state_dict = torch.load(trained_weigths_fname)['state_dict']` to `state_dict = torch.load(trained_weigths_fname, map_location=torch.device('cpu'))['state_dict']`


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
- Download the dji rosbags that you can find [here](https://keeper.mpdl.mpg.de/d/1cae0814c4474f5a8e19/) either from the `test_bag` or `train_bag` folders.
- Download the checkpoint `airpose_asv3_same_hparams_checkpoint.tar.gz` from [here](https://keeper.mpdl.mpg.de/d/1cae0814c4474f5a8e19/) 

In separated terminals (with the workspace sourced) run:
- `roscore`
- `rosparam set use_sim_time true`
- Launch the first client (i.e. the first "drone") with `roslaunch airpose_client one_robot.launch host:=127.0.0.1 port:=9901 feedback_topic:=info img_topic:=camera/image_raw camera_info_topic:=camera/info robotID:=1 reproject:=false groundtruth:=true`, with host you can change the server IP address, port must correspond, `feedback_topic` must contain the ROI and is of type `neural_network_detector::NeuralNetworkFeedback`, robotID should be either 1 or 2, `reproject` is used to avoid a reprojection to different intrisics parameters and `groundtruth:=true` is used to provide `{min_x, max_x, min_y, max_y}` in the ROI message (description below) 
- Launch the second client `roslaunch airpose_client one_robot.launch host:=127.0.0.1 port:=9902 feedback_topic:=info img_topic:=camera/image_raw camera_info_topic:=camera/info robotID:=2 reproject:=false groundtruth:=true`
- Launch the servers, default IP 127.0.0.1

  *Using the virtualenv/python3.8 installations with previous requirements installed*

   - Firstly change folder to `cd AirPose/catkin_ws/src/aircap/packages/flight/airpose_server` 
      - First server, run `python server.py -p 9901 -m /path/to/the/file.ckpt`, note that `-p port` needs to match the client_1 port
      - Second server, run `python server.py -p 9902 -m /path/to/the/file.ckpt`, note that `-p port` needs to match the client_2 port
- To visualize the results you need to install some dependencies 

  *Using the virtualenv/python3.8 installations with previous requirements installed*.

     - run `pip install meshcat rospkg`
     - Change folder to `cd AirPose/catkin_ws/src/aircap/packages/flight/airpose_server` and run `pip install -e smplx`.
     - The visualization node can then be run with `python copenet_rosViz.py /machine_1/step3_pub /absolute/path/to/smplx/models` or `python copenet_rosViz.py /machine_2/step3_pub /absolute/path/to/smplx/models`. The path is most likely `/path/to/AirPose/copenet/src/copenet/data/smplx`
- `rosbag play d*_split1.bag --clock --pause`, where split{n-th} is the n-th split of a longer sequence. The splits have some overlap between them. If your pc is powerful enough you might also want to try the full bags.

At this point you should be able to see play the rosbag in the way you prefer.

The published topics, for each machine, are:
- `machine_x/step1_pub`, the results of the first step of the network, read by the other machine
- `machine_x/step2_pub`, the results of the second step of the network, read by the other machine
- `machine_x/step3_pub`, the final results of `machine_x`

The ROI message can be either used as "grountruth" box with the following structure:
```
ymin = ymin
ymax = ymax
ycenter = xmin
xcenter = xmax
```
Or as a more general box where you specify the center and the height of the box. In that case a 3:4 aspect ratio is considered.
```
ymin = ymin
ymax = ymax
xcenter = x_center_of_the_bb
ycenter = y_center_of_the_bb
```

You can also create your bag and provide your own data to the tool. To that end you can check the code available [here](https://github.com/eliabntt/RosbagFromCsv) that uses a csv with the needed information (image paths, bounding boxes, and camera info) to build the bags.

Note that this is no different than running the inference manually, except for the fact that this runs at 4FPS and has the synchronization procedure enabled as explained in the paper.
