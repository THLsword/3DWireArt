import networkx as nx
import os
import tqdm
import torch
import numpy as np
import argparse

def minimum_path_coverage(graph):
    # curve num = max(1, 奇數degree數量/2)

    print(graph)