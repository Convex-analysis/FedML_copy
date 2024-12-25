#!/usr/bin/env bash

nohup python multi_hierarchicalFLtest.py --cf cifar10_config_0.yaml > cifar10_config_0.log 2>&1 &

nohup python multi_hierarchicalFLtest.py --cf cifar10_config_6.yaml > cifar10_config_6.log 2>&1 &

nohup python multi_hierarchicalFLtest.py --cf cifar10_config_12.yaml > cifar10_config_12.log 2>&1 &

nohup python multi_hierarchicalFLtest.py --cf cifar10_config_18.yaml > cifar10_config_18.log 2>&1 &

nohup python multi_hierarchicalFLtest.py --cf cifar10_config_24.yaml > cifar10_config_24.log 2>&1 &

nohup python multi_hierarchicalFLtest.py --cf cifar10_config_30.yaml > cifar10_config_30.log 2>&1 &

nohup python multi_hierarchicalFLtest.py --cf cifar10_config_36.yaml > cifar10_config_36.log 2>&1 &

nohup python multi_hierarchicalFLtest.py --cf cifar10_config_42.yaml > cifar10_config_42.log 2>&1 &