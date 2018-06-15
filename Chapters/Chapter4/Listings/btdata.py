#!/usr/bin/env python

import json
import networkx as nx
import pygraphviz
import rospy

from cyborg_msgs.msg import BehaviorTree, BehaviorTreeNodes
from networkx.utils.misc import make_str

class BTData(object):
    def __init__(self, data_sub_name=None, update_sub_name=None):
        if data_sub_name:
            rospy.Subscriber(data_sub_name, BehaviorTree, self._bt_cb)

        if update_sub_name:
            rospy.Subscriber(update_sub_name, BehaviorTreeNodes, self._bt_update_cb)

        self.tree = None
        self.active_nodes = []

    def get_graph(self):
        # Create a graph from the JSON data, or fall back to an empty graph
        try:
            graph = tree_graph(self.tree)
        except:
            graph = nx.OrderedDiGraph()

        G = nx.drawing.nx_agraph.to_agraph(graph)

        return G, self.active_nodes

    def _bt_cb(self, msg):
        self.tree = json.loads(msg.tree)

    def _bt_update_cb(self, msg):
        self.active_nodes = msg.ids
