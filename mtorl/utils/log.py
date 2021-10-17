import logging

logger = logging.getLogger('log')
formatter = logging.Formatter(
    '%(asctime)s %(filename)s:%(lineno)d %(levelname)s| %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def set_logger(log_file='log.txt', mode='w'):

    while len(logger.handlers):
        logger.removeHandler(logger.handlers[0])
    logger.setLevel(logging.DEBUG)

    if log_file is not None:
        fh = logging.FileHandler(log_file, mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)


set_logger(log_file=None)
