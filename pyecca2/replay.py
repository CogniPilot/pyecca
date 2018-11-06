import os
import pyulog
from typing import List
from dataclasses import dataclass
import numpy as np
import simpy
from pyecca2 import msgs, uros
from pyecca2.uros import Publisher


@dataclass
class LogEvent:
    timestamp: int
    index: int
    topic: pyulog.ULog.Data


class ULogReplay:

    def __init__(self, core: uros.Core, ulog_file: str):

        with open(ulog_file, 'rb') as f:
            ulog = pyulog.ULog(f)

        # annotate types for python editors
        topic_list = ulog.data_list  # type: List[pyulog.ULog.Data]
        event_list = []

        # sort every publication up front by timestamp, might be a faster way to do this, but
        # very quick during actual sim
        for topic in topic_list:
            for i, timestamp in enumerate(topic.data['timestamp']):
                event_list.append(LogEvent(timestamp=timestamp, index=i, topic=topic))
        event_list.sort(key=lambda x: x.timestamp)

        self.pubs = {}
        for topic in topic_list:
            if topic.name == "sensor_combined":
                self.pubs['imu'] = Publisher(core, 'imu', msgs.Imu)

        # class members
        self.core = core  # type: simpy.Environment
        self.data_list = event_list  # type: List[LogEvent]

        # start process
        self.core.process(self.run())

    def run(self):
        index = 0
        while index < len(self.data_list):
            log = self.data_list[index]  # type: LogEvent
            wait = log.timestamp/1.0e6 - self.core.now
            assert wait >= 0
            yield self.core.timeout(wait)

            t = log.topic.data['timestamp'][log.index]/1.0e6

            # data message
            m = None

            if log.topic.name == "sensor_combined":
                m = msgs.Imu()
                m.data['time'] = t
                m.data['gyro'] = np.array([
                    log.topic.data['gyro_rad[0]'][log.index],
                    log.topic.data['gyro_rad[1]'][log.index],
                    log.topic.data['gyro_rad[2]'][log.index]
                ])
                m.data['accel'] = np.array([
                    log.topic.data['accelerometer_m_s2[0]'][log.index],
                    log.topic.data['accelerometer_m_s2[1]'][log.index],
                    log.topic.data['accelerometer_m_s2[2]'][log.index]
                ])
            elif log.topic.name == "vehicle_magnetometer":
                m = msgs.Mag()
                m.data['time'] = t
                m.data['mag'] = np.array([
                    log.topic.data['magnetometer_ga[0]'][log.index],
                    log.topic.data['magnetometer_ga[1]'][log.index],
                    log.topic.data['magnetometer_ga[2]'][log.index]
                ])
            elif log.topic.name == "vehicle_attitude":
                pass
            elif log.topic.name == "vehicle_attitude_groundtruth":
                pass
            elif log.topic.name == "vehicle_air_data":
                pass
            elif log.topic.name == "vehicle_rates_setpoint":
                pass
            elif log.topic.name == "vehicle_attitude_setpoint":
                pass
            elif log.topic.name == "rate_ctrl_status":
                pass
            elif log.topic.name == "actuator_controls_0":
                pass
            elif log.topic.name == "vehicle_local_position_groundtruth":
                pass
            elif log.topic.name == "vehicle_global_position_groundtruth":
                pass
            elif log.topic.name == "vehicle_actuator_outputs":
                pass
            elif log.topic.name == "vehicle_gps_position":
                pass
            elif log.topic.name == "vehicle_local_position_setpoint":
                pass
            elif log.topic.name == "actuator_outputs":
                pass
            elif log.topic.name == "battery_status":
                pass
            elif log.topic.name == "manual_control_setpoint":
                pass
            elif log.topic.name == "vehicle_land_detected":
                pass
            elif log.topic.name == "telemetry_status":
                pass
            elif log.topic.name == "vehicle_status_flags":
                pass
            elif log.topic.name == "vehicle_status":
                pass
            elif log.topic.name == "sensor_preflight":
                pass
            elif log.topic.name == "vehicle_command":
                pass
            elif log.topic.name == "commander_state":
                pass
            elif log.topic.name == "actuator_armed":
                pass
            elif log.topic.name == "sensor_selection":
                pass
            elif log.topic.name == "estimator_status":
                pass
            else:
                print('unhandled', log.topic.name)

            #if m is not None:
            #    print('publishing', m)

            index += 1
