import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jonurce/Jon/ROS2Projects/idea_0_ws/install/event_reader'
