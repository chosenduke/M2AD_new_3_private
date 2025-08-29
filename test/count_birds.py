#!/usr/bin/env python3
import os

root = "/root/autodl-tmp/mydata/M2AD/Bird"
for dirpath, dirnames, filenames in os.walk(root):
    # 叶子目录: 不含子目录
    if not dirnames:
        rel = os.path.relpath(dirpath, root)
        print(f"{rel}\t{len(filenames)}")