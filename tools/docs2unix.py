#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: python dos2unix.py
"""

import sys

# original = "D:\\ud120-projects\\tools\\word_data.pkl"
# destination = "D:\\ud120-projects\\tools\\word_data_unix.pkl"

original = "D:\\ud120-projects\\final_project\\final_project_dataset.pkl"
destination = "D:\\ud120-projects\\final_project\\final_project_dataset_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))
