# %%
import rosbag
import numpy as np
import matplotlib.pyplot as plt

bag1 = rosbag.Bag("/ps/project/aircap/raw_data/2021/2021-12-03/experiments/therealdeal/machine_1/rosbag_self_1638535198.bag")
bag2 = rosbag.Bag("/ps/project/aircap/raw_data/2021/2021-12-03/experiments/therealdeal/machine_2/rosbag_self_1638535199.bag")

tstamp1 = []
step3_out1 = []
for topic, msg, t in bag1.read_messages(topics=['/machine_1/step3_pub']):
    step3_out1.append(np.array(msg.data).reshape(1,-1))
    tstamp1.append(t.to_nsec())
tstamp1 = np.array(tstamp1)
tstamp1 = (tstamp1-tstamp1[0])/(1e9*60)
step3_out1 = np.concatenate(step3_out1)

trans1 = step3_out1[:,10:13]*20
plt.figure()
plt.plot(trans1[np.logical_and(tstamp1>2, tstamp1<3)])


step3_out2 = []
tstamp2 = []
for topic, msg, t in bag2.read_messages(topics=['/machine_2/step3_pub']):
    step3_out2.append(np.array(msg.data).reshape(1,-1))
    tstamp2.append(t.to_nsec())
tstamp2 = np.array(tstamp2)
tstamp2 = (tstamp2 - tstamp2[0])/(1e9*60)
step3_out2 = np.concatenate(step3_out2)

trans2 = step3_out2[:,10:13]*20
plt.figure()
plt.plot(trans2[np.logical_and(tstamp2>2, tstamp2<3)])

# %%
import rosbag
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

exp = "/ps/project/aircap/nitin_2021_12/flight_1"

bag1 = rosbag.Bag(os.path.join(exp,"m1data.bag"))
bag2 = rosbag.Bag(os.path.join(exp,"m2data.bag"))

step3_out1 = []
tstamp1 = []
cam1_poses = []
person_pose1 = []
for topic, msg, t in bag1.read_messages(topics=['/machine_1/step3_pub',"/machine_1/object_detections/camera_debug","/machine_1/target_tracker/pose"]):
    if topic == '/machine_1/step3_pub':
        step3_out1.append(np.array(msg.data).reshape(1,-1))
        tstamp1.append(t.to_nsec())
    if topic == "/machine_1/object_detections/camera_debug":
        tr = np.array([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z]).reshape(3,1)
        r = (R.from_quat([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w])).as_matrix()
        pose = np.concatenate([r,tr],axis=1)
        cam1_poses.append({str(t.to_nsec()):pose})
    if topic == "/machine_1/target_tracker/pose":
        tr = np.array([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z]).reshape(3,1)
        person_pose1.append({str(t.to_nsec()):tr})


step3_out2 = []
tstamp2 = []
cam2_poses = []
person_pose2 = []
for topic, msg, t in bag2.read_messages(topics=['/machine_2/step3_pub',"/machine_2/object_detections/camera_debug","/machine_2/target_tracker/pose"]):
    if topic == '/machine_2/step3_pub':
        step3_out2.append(np.array(msg.data).reshape(1,-1))
        tstamp2.append(t.to_nsec())
    if topic == "/machine_2/object_detections/camera_debug":
        tr = np.array([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z]).reshape(3,1)
        r = (R.from_quat([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w])).as_matrix()
        pose = np.concatenate([r,tr],axis=1)
        cam2_poses.append({str(t.to_nsec()):pose})
    if topic == "/machine_2/target_tracker/pose":
        tr = np.array([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z]).reshape(3,1)
        person_pose2.append({str(t.to_nsec()):tr})

