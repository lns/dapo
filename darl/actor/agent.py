#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function, unicode_literals)

try:
    from future.builtins import ascii, filter, hex, map, oct, zip
except ImportError:
    pass

import numpy as np
from darl.actor.base_agent import BaseAgent
from darl.memoire import ReplayMemoryClient, get_host_ip, get_pid
from threading import Thread

class Agent(BaseAgent):
    client = None

    def __init__(self, uuid=None, serial=None,
            sub_ep='', req_ep='', push_ep='', push_length=0, **kwargs):
        self.threads = []
        # Memoire
        if uuid is None:
            uuid = "RMC::IP:%s:PID:%d" % (get_host_ip('8.8.8.8', 23), get_pid())
        self.client = ReplayMemoryClient(uuid)
        self.client.sub_endpoint = sub_ep
        self.client.req_endpoint = req_ep
        self.client.push_endpoint = push_ep
        self.client.push_length = push_length
        if serial is None or serial=='':
            print("Initialize model on the wire..")
            assert self.client.sub_endpoint
            serial = self.client.sub_bytes('darl.model')
        else:
            print("Loading from saved model..")
        super(Agent, self).__init__(serial=serial, **kwargs)
        if self.client.req_endpoint:
            self.client.get_info()
        # SyncManager
        if self.client.sub_endpoint:
            self.threads.append(Thread(target=self.update_agent))
        if self.client.push_endpoint:
            self.threads.append(Thread(target=self.client.push_worker_main))
        # start
        for th in self.threads:
            th.daemon = True # TODO: do we need this daemon feature?
            th.start()

    def check(self):
        # TODO: some shapes and dtypes should match
        for k,v in self.nodes.items():
            print("node['%s']: %s %s" % (k, str(v.dtype), str(v.shape)))
        for i in range(self.client.view_size):
            print(self.client.view(i))

    def __del__(self):
        self.client.close()

    def add_entry(self, entry, term):
        """
        User friendly version of self.client.add_entry(entry, term).
        Automatically convert data type to desired.
        """
        tpl = self.client.template
        if len(entry) != len(tpl):
            print(entry)
            for i in range(len(tpl)):
                print(str(tpl[i].dtype) + ' : ' + str(tpl[i].shape))
        assert len(entry) == len(tpl)
        for i in range(len(entry)):
            entry[i] = np.asarray(entry[i], dtype=tpl[i].dtype)
        entry = tuple(entry)
        self.client.add_entry(entry, term)

    def update_agent(self):
        try:
            while True:
                serial = self.client.sub_bytes('darl.model')
                print('Received model message of size %d.' % len(serial))
                self.mutex.acquire()
                self.deserialize(serial)
                self.mutex.release()
        except KeyboardInterrupt:
            pass

