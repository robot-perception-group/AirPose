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
Please download the data, and untar it whenever necessary. Content details are following:

- copenet_synthetic_data.tar.gz - Synthetic dataset
- copenet_dji_real_data.tar.gz - real dataset
- hmr_synthetic.tar.gz - Baseline pretrained checkpoint on synthetic data and pkl files for precalculated results
- copenet_singleview_ckpt.zip - Baseline+Fullcam pretrained checkpoint on synthetic data and pkl files for precalculated results
- muhmr_synthetic.tar.gz - Baseline+Multiview pretrained checkpoint on synthetic data and pkl files for precalculated results
- copenet_twoview_synthetic_ckpt.tar.gz - AirPose pretrained checkpoint on synthetic data and pkl files for precalculated results
- hmr_real_ckpt.zip - Baseline finetuned checkpoint on real data and pkl files for precalculated results
- copenet_twoview_real_ckpt.zip - AirPose finetuned checkpoint on real data and pkl files for precalculated results
- SMPLX_to_J14.pkl - Mapping from SMPLX joints to the 14 joints format of openpose. It is used in the method AirPose+.
_________

The code was tested using `Python 3.8`.

`SMPLX` submodule in this repo is a modified version of the official SMLX [implementation](https://github.com/vchoutas/smplx). Download the SMPLX model weights from [here](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip) and run the following

```
# from the download location
unzip models_smplx_v1_1.zip -d models_smplx
unzip models_smplx/models/smplx/smplx_npz.zip -d used_models
rm models_smplx -r
```

Then copy the *content* of `used_models` (just created, with `SMPLX_{MALE,FEMALE,NEUTRAL}.npz` files) folder into `your_path/AirPose/copenet/src/copenet/data/smplx/models/smplx`.

You need to register before being able to download the weights.

Now, you may want to create a virtual environment. Please be sure your `pip` is updated.

Install the necessary requirements with `pip install -r requirements.txt`. If you don't have a cuda compatible device, change the device to `cpu` in `copenet_real/src/copenet_real/config.py` and `copenet/src/copenet/config.py`. 

In those files (`copenet_real/src/copenet_real/config.py` and `copenet/src/copenet/config.py`) change `LOCAL_DATA_DIR` to `/global/path/AirPose/copenet/src/copenet/data"`.

Check out [this](https://stackoverflow.com/questions/65637222/runtimeerror-subtraction-the-operator-with-a-bool-tensor-is-not-supported) link to fix the runtime error `RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported` due to the `Torchgeometry` package.

Install the copenet and copenet_real packages in this repo
```
pip install -e copenet
pip install -e copenet_real
```

Download the head and hands indices files form [here](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_mano_flame_correspondences.zip) and place them in `AirPose/copenet/src/copenet/data/smplx` (`MANO_SMPLX_vertex_ids.pkl` and `SMPL-X__FLAME_vertex_ids.npy`).

## Synthetic data training 

The data to be used is `copenet_synthetic_data.tar.gz` ([here](https://keeper.mpdl.mpg.de/d/1cae0814c4474f5a8e19/files/?p=%2Fcopenet_synthetic_data.tar.gz))

### Preprocess
To run the code of this repository you first need to preprocess the data using

```
# from AirPose folder
python copenet/src/copenet/scripts/prepare_aerialpeople_dataset.py /absolute/path/copenet_synthetic
```

And code can be run by the following (from `AirPose/copenet` folder):

`python src/copenet/copenet_trainer.py --name=test_name --version=test_version --model=muhmr --datapath=/absolute/path/copenet_synthetic --log_dir=path/location/ --copenet_home=/absolute/path/AirPose/copenet --optional-params...`

The `datapath` is the location of the training data.

`--model` specify the model type between `[hmr, muhmr, copnet_singleview, copenet_twoview]` which corresponds to the Baseline, Baseline+multi-view, Baseline+Fullcam and AirPose respectively.

Logs will be saved in `$log_dir/$name/$version/`

`optional-params` is to be substituted with the `copenet_trainer` available params as weights, lr..

## Evaluation on the synthetic data

For model type `[muhmr, copenet_twoview]`.

```
cd AirPose/copenet_real

python src/copenet_real/scripts/copenet_synth_res_compile.py "model type" "checkpoint Path" "/path to the dataset"
```

For model type `[hmr, copenet_singleview]`, the provided checkpoint is trained with an older pytorch lightning version (<=1.2). If you want to use them, install pytorch-lightning<=1.2. 

We provide the precalculated outputs on the syntehtic data using these checkpoints. 

To generate the metrics, run
```
cd AirPose/copenet_real

python src/copenet_real/scripts/hmr_synth_res_compile.py "model type" "precalculated results directory Path" "/path to the dataset" "your_path/AirPose/copenet/src/copenet/data/smplx/models/smplx"
```

## Fine-tuning on real dataset
The data to be used is `copenet_dji_real_data.tar.gz`([here](https://keeper.mpdl.mpg.de/d/1cae0814c4474f5a8e19/files/?p=%2Fcopenet_dji_real_data.tar.gz)).

Install the human body prior from [here](https://github.com/nghorbani/human_body_prior) and download its pretrained weights (version 2) from [here](https://smpl-x.is.tue.mpg.de/download.php). Set the `vposer_weights` variable in the `.../AirPose/copenet_real/src/copenet_real/config.py` file to the absolute path of the downloaded weights (e.g. `/home/user/Downloads/V02_05`). If you do NOT have a GPU please change `human_body_prior/tools/model_loader.py` line **68** from `state_dict = torch.load(trained_weigths_fname)['state_dict']` to `state_dict = torch.load(trained_weigths_fname, map_location=torch.device('cpu'))['state_dict']`

**Note**: for the `hmr` (Baseline) model `pytorch-lightning<=1.2` is required. You might have to recheck requirements, or reinstall the requirements you can find in the main folder of this repo.

Code can be run by the following (from `AirPose/copenet_real/` folder)

```
python src/copenet_real/copenet_trainer.py --name=test_name --version=test_version --model=hmr --datapath=path/location --log_dir=path/location/ --resume_from_checkpoint=/path/to/checkpoint --copenet_home=/absolute/path/AirPose/copenet --optional-params...
```

The `datapath` is the location of the training data.

`--model` specify the model type between `[hmr, copenet_twoview]` which corresponds to the Baseline, AirPose respectively.

The `--resume_from_checkpoint` is path to the pretrained checkpoint on the synthetic data. 

## Evaluation on real data

Install `graphviz` dependency with `pip install graphviz` in the same virtual environment.

Following code will generate the plots comparing the results of the baseline method, AirPose and AirPose+ on the real data. 

This can be run from `AirPose` folder.

```
python copenet_real_data/scripts/bundle_adj.py "path_to_the_real_dataset" \\
"path_to_the_SMPLX_neutral_npz_file" \\
"path_to_vposer_folder" \\
"path_to_the_hmr_checkpoint_directory" \\
"path_to_the_airpose_precalculated_res_on_realdata_pkl" \\
"path_to_the_SMPLX_to_j14_mapping_pkl_file" \\
"type_of_data(train/test)" 

```
Note that:
- The `SMPLX_neutral_npz_file` should be in `your_path/AirPose/copenet/src/copenet/data/smplx/models/smplx`.
- The `vposer_folder` should be in the `vposer_weights` folder that you downloaded to finetune on the real data
- The hmr checkpoint is either being generated by you or downloaded from [here](https://keeper.mpdl.mpg.de/d/1cae0814c4474f5a8e19/files/?p=%2Fhmr_real_ckpt.zip)
- The `precalculated_res_on_realdata_pkl` can be found within the same archive you downloaded above. More on how to compute them yourself below.
- The `SMPLX_to_j14_pkl` can be found [here](https://keeper.mpdl.mpg.de/d/1cae0814c4474f5a8e19/files/?p=%2FSMPLX_to_J14.pkl).

The evaluation code above needs precalculated results on the real data which are provided with the dataset. If you want to calculate them yourself, run the following code and save the variable `outputs` in a pkl file when a breakpoint is hit. The pkl files provided with the data are generated in the same way.
For `AirPose`
```
python copenet_real/src/copenet_real/scripts/copenet_real_res_compile.py "checkpoint Path" "/path to the dataset"
```
For `Baseline`
```
python copenet_real/src/copenet_real/scripts/hmr_real_res_compile.py "checkpoint Path" "/path to the dataset"
```

## Testing the client-server synchronization mechanism

To this end you need to install `ros-{melodic,noetic}` in your pc (`Ubuntu 18.04-20.04`).

Please follow the instructions that you can find [here](http://wiki.ros.org/Installation/Ubuntu)

After that you need to install the following dependencies:

```
sudo add-apt-repository ppa:joseluisblancoc/mrpt-stable
```

Navigate to your `catkin_ws` folder (e.g. `AirPose/catkin_ws`) and run:

```
touch src/aircap/packages/optional/basler_image_capture/Grab/CATKIN_IGNORE
touch src/aircap/packages/optional/ptgrey_image_capture/Grab/CATKIN_IGNORE
```

### this applies to ros-melodic

Firstly, checkout the AirPose **branch** `ros-melodic`.

Be *sure* to update the submodule (first command).

```
git submodule update 
sudo apt install libmrpt-dev mrpt-apps
cd /your/path/AirPose/catkin_ws
touch src/aircap/packages/3rdparty/mrpt_bridge/CATKIN_IGNORE
touch src/aircap/packages/3rdparty/pose_cov_ops/CATKIN_IGNORE
sudo apt install -y ros-melodic-octomap-msgs ros-melodic-cv-camera ros-melodic-marker-msgs ros-melodic-mrpt-msgs ros-melodic-octomap-ros ros-melodic-mrpt-bridge ros-melodic-mrpt1
```

### this applies to ros-noetic
```
sudo apt install libmrpt-poses-dev libmrpt-obs-dev libmrpt-graphs-dev libmrpt-maps-dev libmrpt-slam-dev -y
sudo apt install -y ros-noetic-octomap-msgs ros-noetic-cv-camera ros-noetic-marker-msgs ros-noetic-mrpt-msgs ros-noetic-octomap-ros ros-noetic-mrpt2
```

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
- You can either use the complete bag files with
  `rosbag play d*_BGR.bag --clock --pause`
  
  or create smaller (overlapping) bags using the `split.zsh` script that you find in both folders. This split will create 5 split from each bag. Afterwards, simply run `rosbag play d*_split1.bag --clock --pause`, where split{n-th} is the n-th split of the longer sequence. The splits have some overlap between them.
  
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
