import datetime
import os
import re
import sys
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from loguru import logger

from utils.utils import is_dist_avail_and_initialized


class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count else 0.0

    @property
    def max(self):
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self):
        return self.deque[-1] if self.deque else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter=", ", log_file=""):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = Logger(log_file=log_file) if log_file else Logger.disabled()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (float, int)):
                self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        total_len = len(iterable)
        space_fmt = ":" + str(len(str(total_len))) + "d"
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total_len - 1:
                eta_seconds = iter_time.global_avg * (total_len - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_str = [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
                if torch.cuda.is_available():
                    log_str.append("max mem: {memory:.0f}")
                message = self.delimiter.join(log_str).format(
                    i,
                    total_len,
                    eta=eta_string,
                    meters=str(self),
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB if torch.cuda.is_available() else 0,
                )
                self.logger.info(message)
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / max(total_len, 1)
            )
        )


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        is_ddp = is_dist_avail_and_initialized()
        current_rank = dist.get_rank() if is_ddp else 0
        self.is_main_process = current_rank == 0

        if not self.is_main_process:
            return

        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}",
            level="INFO",
            colorize=True,
        )
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logger.add(log_file, format=self._file_formatter, level="INFO", rotation="10MB")
            logger.info(f"Logging to {log_file}")

    @classmethod
    def disabled(cls):
        obj = cls.__new__(cls)
        obj.log_file = ""
        obj.is_main_process = True
        return obj

    def write(self, message):
        if message.strip():
            self.info(message.strip())

    def flush(self):
        pass

    @staticmethod
    def info(msg):
        logger.info(msg)

    @staticmethod
    def warning(msg):
        logger.warning(msg)

    @staticmethod
    def error(msg):
        logger.error(msg)

    @staticmethod
    def _strip_ansi(text):
        ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
        return ansi_re.sub("", text)

    def _file_formatter(self, record):
        time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S")
        message = self._strip_ansi(record["message"])
        return f"{time_str} | {message}\n"
