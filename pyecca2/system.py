import simpy
import pyecca2.msgs as msgs
import numpy as np
import copy


class Core(simpy.Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._params = None
        self._publishers = {}
        self._subscribers = {}
        self._declared_params = {}
        self.pub_sub_locked = False
        self.pub_params = Publisher(self, 'params', msgs.Params)

    def get_param(self, name):
        return self._params.data[name]

    def set_param(self, name, value):
        self._params.data[name] = value
        self.pub_params.publish(self._params)

    def declare_param(self, param):
        assert not self.pub_sub_locked
        assert param.name not in self._declared_params
        self._declared_params[param.name] = param

    def run(self, *args, **kwargs):
        self._params = msgs.Params(self)
        self.pub_params.publish(self._params)
        super().run(*args, **kwargs)


class Subscriber:

    def __init__(self, core, topic, msg_type, callback):
        assert not core.pub_sub_locked
        self.core = core
        self.topic = topic
        self.msg_type = copy.deepcopy(msg_type)
        self.callback = callback
        if topic not in core._subscribers:
            core._subscribers[topic] = []
        core._subscribers[topic].append(self)


class Publisher:

    def __init__(self, core, topic, msg_type):
        assert not core.pub_sub_locked
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
        self.pub_sub_locked = False


class Param:

    def __init__(self, core, name, value, dtype):
        assert not core.pub_sub_locked
        self.core = core
        self.name = name
        self.value = value
        self.dtype = dtype
        core.declare_param(self)

    def set(self, value):
        self.value = value
        self.core._set_param(self.name, value)

    def get(self):
        return self.value

    def update(self):
        self.value = self.core.get_param(self.name)


class Logger:

    def __init__(self, core):
        self.core = core
        self.dt = Param(core, 'logger/dt', 1.0/200, 'f4')
        self.data_latest = None
        self.data_list = []
        simpy.Process(core, self.run())
        self.subs = {}
        for topic, publisher in self.core._publishers.items():
            cb = lambda msg, topic=topic: self.callback(topic, msg)
            self.subs[topic] = Subscriber(
                self.core, topic, publisher.msg_type, cb)
        self.data_latest = msgs.Log(self.core)
        self.core.pub_sub_locked = True
        self.param_list = [self.dt]

    def callback(self, topic, msg):
        self.data_latest.data[topic] = copy.deepcopy(msg.data)
        if topic == 'params':
            for p in self.param_list:
                p.update()

    def run(self):

        while True:
            self.data_latest.data['time'] = self.core.now
            self.data_list.append(copy.deepcopy(self.data_latest.data))
            yield simpy.Timeout(self.core, self.dt.get())

    def get_log_as_array(self):
        return np.array(self.data_list, dtype=self.data_latest.dtype)

