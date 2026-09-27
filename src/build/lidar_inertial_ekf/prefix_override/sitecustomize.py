import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/sarath/ekf-pose-fusion/src/install/lidar_inertial_ekf'
