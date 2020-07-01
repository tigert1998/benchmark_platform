# Network

This folder contains the implementation for various neural network structures.
All code are purely based on TensorFlow 1 for compatibility on edge devices.
The code structure has two levels: operators and blocks.

|Level|File(s)|Remark|
|---|---|---|
|Operators|[building_ops.py](building_ops.py)|It contains implementation for multiple operators, such as channel shuffle, group conv.|
|Blocks|[dense_blocks.py](dense_blocks.py), [mbnet_blocks.py](mbnet_blocks.py), [resnet_blocks.py](resnet_blocks.py), [shufflenet_units.py](shufflenet_units.py)|Each file contains corresponding code for specific block. Some involving operators are defined in [building_ops.py](building_ops.py).|
