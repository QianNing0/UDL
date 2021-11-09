import sys
import threading
if sys.version > '3':
    import queue as Queue
else:
	import Queue
import random
import collections

import torch
import torch.multiprocessing as multiprocessing

from torch._C import _set_worker_signal_handlers
from torch.utils.data import _utils
from torch.utils.data._utils.pin_memory import _pin_memory_loop
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter
_use_shared_memory = False

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, init_fn, worker_id):
    global _use_shared_memory
    _use_shared_memory = True
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        try:
            idx_scale = 0
            if len(scale) > 1 and dataset.train:
                idx_scale = random.randrange(0, len(scale))
                dataset.set_scale(idx_scale)

            samples = collate_fn([dataset[i] for i in batch_indices])
            samples.append(idx_scale)

        except Exception:
            data_queue.put((idx, _utils.ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


class MSDataLoader(DataLoader):
    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        collate_fn=_utils.collate.default_collate,
            pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None):
        super(MSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=args.n_threads, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)

        self.scale = args.scale

    def __iter__(self):
        return _MultiProcessingDataLoaderIter(self)


