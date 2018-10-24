import simpy
import examples.msgs as msgs
import numpy as np
import copy
import time


class Core(simpy.Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._params = msgs.Params()
        self._publishers = {}
        self._subscribers = {}
        self.pub_params = Publisher(self, 'params', msgs.Params)

    def set_params(self, params):
        for k, v in params.items():
            self._params.data[k] = v

    def _get_param(self, name):
        return self._params.data[name]

    def _set_param(self, name, value):
        self._params.data[name] = value
        self.pub_params.publish(self._params)

    def run(self, *args, **kwargs):
        self.pub_params.publish(self._params)
        super().run(*args, **kwargs)


class Subscriber:

    def __init__(self, core, topic, msg_type, callback):
        self.core = core
        self.topic = topic
        self.msg_type = copy.deepcopy(msg_type)
        self.callback = callback
        if topic not in core._subscribers:
            core._subscribers[topic] = []
        core._subscribers[topic].append(self)


class Publisher:

    def __init__(self, core, topic, msg_type):
        self.core = core
        self.topic = topic
        self.msg_type = msg_type
        assert topic not in core._publishers
        core._publishers[topic] = self

    def publish(self, msg):
        if not isinstance(msg, self.msg_type):
            raise ValueError("{:s} expects msg {:s}, but got {:s}".format(
                self.topic, str(self.msg_type), str(type(msg))))
        if self.topic not in self.core._subscribers:
            return
        for s in self.core._subscribers[self.topic]:
            s.callback(msg)


class Param:

    def __init__(self, core, name, value):
        self.core = core
        self.name = name
        self.value = value
        core._set_param(name, value)

    def set(self, value):
        self.value = value
        self.core._set_param(self.name, value)

    def get(self):
        return self.value

    def update(self):
        self.value = self.core._get_param(self.name)




class Logger:

    def __init__(self, core):
        self.core = core
        self.dt = Param(core, 'logger/dt', 0.1)
        self.subs = {}

        for topic, publisher in core._publishers.items():
            cb = lambda msg, topic=topic: self.callback(topic, msg)
            self.subs[topic] = Subscriber(
                core, topic, publisher.msg_type, cb)
        self.data_latest = msgs.Log(core)
        self.data_list = []
        simpy.Process(core, self.run())

    def callback(self, topic, msg):
        self.data_latest.data[topic] = copy.deepcopy(msg.data)
        if topic == 'params':
            self.dt.update()

    def run(self):
        while True:
            self.data_latest.data['time'] = self.core.now
            self.data_list.append(copy.deepcopy(self.data_latest.data))
            yield simpy.Timeout(self.core, self.dt.get())

    def get_log_as_array(self):
        return np.array(self.data_list, dtype=self.data_latest.dtype)

