#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import (absolute_import, division, print_function, unicode_literals)

try:
    from future.builtins import ascii, filter, hex, map, oct, zip
except ImportError:
    pass
import sys

if sys.version_info.major > 2:
    xrange = range

import numpy as np
import os, time
from darl.memoire import Proxy, Bind, Conn
from threading import Thread
from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_enum("proxy_type", None, ['PubSub', 'ReqRep', 'PushPull'], '?', case_sensitive=False)
flags.DEFINE_string("front_ep", "", "Front Endpoint")
flags.DEFINE_string("back_ep", "", "Back Endpoint")

def main(unused_argv):
    with open("config.txt", "w") as f:
        f.write(FLAGS.flags_into_string())
    proxy = Proxy()

    proxy_type = FLAGS.proxy_type
    front_ep = FLAGS.front_ep
    back_ep = FLAGS.back_ep
    print("Starting proxy: '%s': '%s' -> '%s'" % (proxy_type, front_ep, back_ep))

    try:
        if proxy_type == 'PubSub':
            proxy.pub_proxy_main(front_ep, Conn, back_ep, Bind, 32)
        elif proxy_type == 'ReqRep':
            proxy.rep_proxy_main(front_ep, Bind, back_ep, Conn, 32)
        elif proxy_type == 'PushPull':
            proxy.pull_proxy_main(front_ep, Bind, back_ep, Conn, 32)
        else:
            print("Unknown proxy_type: '%s'" % proxy_type)
    except KeyboardInterrupt:
        pass
    os.kill(os.getpid(), 9)

if __name__ == '__main__':
    app.run(main)

