import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from event_camera_py import Decoder
from event_camera_msgs.msg import EventPacket
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# topic list from rosbags
# /event_camera/events
# /event_camera/renderer/image_raw
# /event_camera/renderer/image_raw/compressed
# /event_camera/renderer/image_raw/compressedDepth
# /events/read_split
# /events/write_split
# /zed/depth/depth_registered
# /zed/rgb/image_rect_color

class EventSubscriber(Node):

    def __init__(self):
        super().__init__('event_subscriber')

        # Store (x, y, p, t) events in an array
        self.events = []

        # Subscriber: read events and put them inside self.events
        self.subscription = self.create_subscription(
            EventPacket,
            '/event_camera/events',
            self.read_events,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        # Service: number_of_events inside self.events
        self.create_service(
            Trigger,
            'number_of_events',
            self.events_size
        )

        # Publisher: event_pointcloud
        # self.cloud_pub = self.create_publisher(
            #    PointCloud2,
            #    '/event_pointcloud',
            #    10
        # )
        # self.timer = self.create_timer(10.0, self.publish_pointcloud)

        # Service: self.events to csv
        self.create_service(
            Trigger,
            'events_to_csv',
            self.events_to_csv
        )

        # Service: csv to plot
        self.create_service(
            Trigger,
            'events_plot_from_csv',
            self.events_plot
        )


    # Function for subscriber: read events and paste them in self.events
    def read_events(self, msg):
        decoder = Decoder()
        decoder.decode(msg)
        cd_events = decoder.get_cd_events()
        self.events.extend([event for event in cd_events])
        #for event in cd_events:
        #    x, y, p, t = event
        #   self.events.append((x, y, p, t))

    # Function for service: get self.events current lenght (number of events)
    def events_size(self, request, response):
        response.success = True
        response.message = f'Total number of events: {len(self.events)}'
        return response

    # Function for publisher: publish events in self.events as a pointcloud
    def publish_pointcloud(self):
        last_events = self.events[-10000:]
        x = np.array([e[0] for e in last_events], dtype=np.float32)
        y = np.array([e[1] for e in last_events], dtype=np.float32)
        p = np.array([e[2] for e in last_events], dtype=np.float32)
        t = np.array([e[3]/1000000 for e in last_events], dtype=np.float32)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        points = np.column_stack((x, y, t, p))

        cloud = PointCloud2(
            header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'),
            height=1,
            width=len(last_events),
            fields=fields,
            is_bigendian=False,
            point_step=16,  # 4 floats * 4 bytes
            row_step=16 * len(last_events),
            data=points.tobytes(),
            is_dense=True
        )

        self.cloud_pub.publish(cloud)

    # Function for service: get self.events and paste events into csv
    def events_to_csv(self, request, response):
        sorted_events = sorted(self.events, key=lambda x: x[3])
        csv_file = 'events.csv'
        try:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'p', 't'])
                for event in sorted_events:
                    writer.writerow(event)
            response.success = True
            response.message = f'Events written to {csv_file}'
        except Exception as e:
            response.success = False
            response.message = f'Failed to write CSV: {str(e)}'
        return response

    # Function for service: get events from csv and plot it
    def events_plot(self, request, response):
        try:
            df = pd.read_csv('events.csv')
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            colors = {0: 'blue', 1: 'red'}
            ax.scatter(df['t'], df['x'], df['y'], c=df['p'].map(colors))
            ax.set_xlabel('T')
            ax.set_ylabel('X')
            ax.set_zlabel('Y')
            ax.set_ylim(0, 1280)
            ax.set_zlim(0, 720)
            plt.title('3D Event Plot')
            plt.show()
            response.success = True
            response.message = f'Plot success'
        except Exception as e:
            response.success = False
            response.message = f'Failed to plot: {str(e)}'
        return response




def main(args=None):
    rclpy.init(args=args)

    event_subscriber = EventSubscriber()

    rclpy.spin(event_subscriber)

    event_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()