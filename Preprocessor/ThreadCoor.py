import tensorflow as tf
import numpy as np
import threading
import time


def MyLoop(coords, worker_id):
    while not coords.should_stop():
        if np.random.rand() < 0.1:
            print('Stoping from id : %d\n' % worker_id)
            coords.request_stop()
        else:
            print('Working on id : %d\n' % worker_id)
        time.sleep(1)


coords = tf.train.Coordinator()
threads = [threading.Thread(target=MyLoop, args=(coords, i,)) for i in range(5)]

for t in threads:
    t.start()

coords.join(threads)
