import re
from collections import defaultdict
from typing import List


def subtokenizer(identifier: str)-> List[str]:
    # Tokenizes code identifiers
    splitter_regex = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    identifiers = re.split('[._\-]', identifier)
    subtoken_list = []

    for identifier in identifiers:
        matches = splitter_regex.finditer(identifier)
        for subtoken in [m.group(0) for m in matches]:
            subtoken_list.append(subtoken)

    return subtoken_list


def extract_path(edges, edge_type):
    next_node_dict = defaultdict(lambda: None)
    for edge in edges:
        if edge[0] == edge_type:
            next_node_dict[edge[1]] = edge[2]
            
    start_node_set = set(next_node_dict.keys()) - set(next_node_dict.values()) 
    curr_node = start_node_set.pop()
    path = []
    while curr_node is not None:
        path.append(curr_node)
        curr_node = next_node_dict[curr_node]
    return path
