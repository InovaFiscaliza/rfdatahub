#!/bin/sh
set -eu

if ! getent group 1008 > /dev/null; then
    groupadd --gid 1008 cdr
fi

usermod --append --groups 1008 dev
getent group 1008
id dev
