# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import gzip
import logging

def read_gzip_stream(path):
    with gzip.open(path, 'rt', encoding="UTF-8") as f:
        for line in f:
            yield line

def read_text_stream(path):
    with open(path, 'r', encoding="UTF-8") as f:
        for line in f:
            yield line

def read(path):
    if path.endswith(".gz"):
        logging.info("Reading GZIP file")
        return read_gzip_stream(path)
    else:
        return read_text_stream(path)
