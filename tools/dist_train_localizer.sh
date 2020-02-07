#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"

$PYTHON -m torch.distributed.launch --nproc_per_node=$2 --master_port=$PORT $(dirname "$0")/train_localizer.py $1 --launcher pytorch ${@:3}
