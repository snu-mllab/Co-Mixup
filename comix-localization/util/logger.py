import logging
import datetime
from util.ioparser import mkdir, parent


class Logger:
    def __init__(self):
        self.lgs = []

    def add_console_logger(self, name, filter=logging.DEBUG):
        lg = logging.getLogger(name)
        lg.setLevel(filter)
        lg = self._clean_handler(lg)
        lg = self._add_console_handler(lg)
        self.lgs.append(lg)

    def add_file_logger(self, name, path, filter=logging.DEBUG):
        mkdir(parent(path))
        lg = logging.getLogger(name)
        lg.setLevel(filter)
        lg = self._clean_handler(lg)
        # lg = self._add_console_handler(lg)
        lg = self._add_file_handler(lg, path)
        self.lgs.append(lg)

    def _clean_handler(self, lg):
        lg.handlers = []
        return lg

    def _add_console_handler(self, lg):
        h = logging.StreamHandler()
        f = logging.Formatter("%(message)s")
        h.setFormatter(f)
        lg.addHandler(h)
        return lg

    def _add_file_handler(self, lg, path):
        h = logging.FileHandler(path)
        f = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        h.setFormatter(f)
        lg.addHandler(h)
        return lg

    def debug(self, msg):
        for lg in self.lgs:
            lg.debug(msg)

    def info(self, msg):
        for lg in self.lgs:
            lg.info(msg)

    def warn(self, msg):
        for lg in self.lgs:
            lg.warn(msg)

    def error(self, msg):
        for lg in self.lgs:
            lg.error(msg)


def get_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def print_args(args, lg):
    lg.info("# ======================================================== #")
    lg.info("# args")
    lg.info("# ======================================================== #")
    size_key = max([len(elem) for elem in vars(args).keys()])
    format = "{:>%d}: {}" % (size_key)
    for k, v in vars(args).items():
        lg.info(format.format(k, str(v)))
