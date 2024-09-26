#!/usr/bin/env python

import json
import sys

sigmf_file = sys.argv[1]
label_class = sys.argv[2]

labels = 0
correct_labels = 0
with open(sigmf_file) as f:
    meta = json.loads(f.read())
    for annotation in meta["annotations"]:
        label = annotation["core:label"]
        labels += 1
        if label == label_class:
            correct_labels += 1

if not labels:
    print("no labels found!")
    sys.exit(-1)

prop = round(correct_labels / labels * 100, 2)
print(correct_labels, labels, prop)
if prop < 99:
    print("predicted class is less than threshold")
    sys.exit(-1)
