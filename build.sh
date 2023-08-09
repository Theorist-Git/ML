#!/usr/bin/sh

set -xe

clang -Wall -Wextra -o $1 $1.c -lm
./$1
