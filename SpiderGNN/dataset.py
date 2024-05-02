import dgl
from dgl.data import DGLDataset

import torch as th
import pandas as pd
from transformers import AutoTokenizer
from tree_sitter_languages import get_parser
import networkx as nx

import random

random.seed(0)

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
parser = get_parser("python")

def get_functions(code):
    # Return list of function nodes
    # name, code, node
    tree = parser.parse(code.encode())
    queue = [tree.root_node]
    while len(queue) > 0:
        node = queue.pop(0)
        if node.type == "function_definition":
            node_data = {
                "name": node.child_by_field_name("name").text.decode(),
                "code": node.text.decode(),
                "node": node,
            }
            yield node_data
        queue.extend(node.children)

def get_calls(function_definition):
    # TODO: Return list of calls inside the function
    # called_name, node
    queue = [function_definition]
    while len(queue) > 0:
        node = queue.pop(0)
        if node.type == "call":
            node_data = {
                "called_name": node.child_by_field_name("function").text.decode(),
                "code": node.text.decode(),
                "node": node,
            }
            yield node_data
        queue.extend(node.children)


class CallGraphDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="synthetic")

    def process(self):
        # TODO: Load a real dataset
        examples = [{"code": """def foo():
    bar()

def bar():
    baz()

def baz():
    foo()
"""}]
        examples = examples * 10

        graphs = []
        for example in examples:
            nx_graph = nx.DiGraph()
            functions = list(get_functions(example["code"]))
            # print("Functions:", functions)
            sequential_id = 0
            functions_to_ids = {}
            functions_by_name = {n["name"]: n for n in functions}
            for func in functions:
                calls = list(get_calls(func["node"]))
                # print("Calls:", calls)
                for call in calls:
                    calling_function = func
                    called_function = functions_by_name[call["called_name"]]
                    calling_id = functions_to_ids.get(calling_function["name"], None)
                    if calling_id is None:
                        calling_id = sequential_id
                        functions_to_ids[calling_function["name"]] = calling_id
                        nx_graph.add_node(calling_id, code=calling_function["code"])
                        sequential_id += 1
                    called_id = functions_to_ids.get(called_function["name"], None)
                    if called_id is None:
                        called_id = sequential_id
                        functions_to_ids[called_function["name"]] = called_id
                        nx_graph.add_node(called_id, code=called_function["code"])
                        sequential_id += 1
                    nx_graph.add_edge(calling_id, called_id)
            function_texts = [code for n, code in nx_graph.nodes(data="code")]
            # print("Texts:", function_texts)
            function_toks = tokenizer(function_texts, padding=True, return_tensors="pt")
            # print(len(nx_attrs), [len(x) for x in nx_attrs])
            # print("Attributes:", function_toks)
            g = dgl.from_networkx(nx_graph)
            g.ndata["input_ids"] = function_toks.input_ids
            g.ndata["attention_mask"] = function_toks.attention_mask
            graphs.append(g)
        self.graphs = graphs

        # Convert the label list to tensor for saving.
        # TODO: Load real labels :)
        labels = [random.choice([0,1]) for _ in range(len(graphs))]
        self.labels = th.LongTensor(labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


if __name__ == "__main__":
    dataset = CallGraphDataset()
    graph, label = dataset[0]
    print(graph, label)
    print("input_ids:")
    print(graph.ndata["input_ids"])
