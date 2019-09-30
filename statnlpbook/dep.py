#!/usr/bin/env python3


def load_conllu(file_path):
    """
    Load a given CoNLL-U file and return a sequence of Python object or dict representing parsed sentences.
    Args:
        file_path: The file to read from, e.g. data/ud/en_ewt-ud-train.conllu or data/ud/da_ddt-ud-dev.conllu
    Returns:
        A sequence of Python object or dict representing parsed sentences.
    """
    if isinstance(file_path, str):
        return load_conllu_lines(file_path.splitlines())
    with open(file_path, encoding="utf-8") as f:
        return load_conllu_lines(f, file_path)


def load_conllu_lines(f, file_path=""):
    trees = []
    tree = None
    for line_no, line in enumerate(list(f) + [""]):  # Append empty line to handle last
        try:
            line = line.strip()
            if tree is None:  # Creating a new tree
                root = {"index": "0", "form": "ROOT"}
                tree = {"nodes": [root]}  # Initialize node list
            if not line:  # Empty line
                if tree and len(tree["nodes"]) > 1:  # Finalize tree and add to list
                    trees.append(tree)
                    tree = None
            elif line.startswith("#"):  # Handle sent_id and text comments
                key, _, value = line[1:].strip().partition(" = ")
                tree[key] = value
            else:  # Read columns into new node
                node = {}
                node["index"], node["form"], node["lemma"], node["upos"], \
                node["xpos"], _, node["head"], node["deprel"], _, _ = line.split("\t")
                if not {".", "-"}.intersection(node["index"]):  # Skip special nodes
                    tree["nodes"].append(node)
        except:
            raise IOError("Failed loading line %d in '%s':\n%s" % (line_no, file_path, line))
    return trees


def arcs_tokens(tree):
    arcs = {(int(node["head"]), int(node["index"]), node["deprel"]) for node in tree["nodes"][1:]}
    tokens = [node["form"] for node in tree["nodes"]]
    return arcs, tokens


def load_arcs_tokens(lines):
    return arcs_tokens(load_conllu(lines)[0])


def save_to_conllu(data, file_path):
    """
    Save a Python object or dict representing a sequence of parsed sentence in back to a .conllu file.
    Args:
        data: The parsed sentences.
        datadir: The file to save to, e.g. data/ud/predictions.conllu
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for tree in data:
            for key, value in tree.items():  # Write comments
                if key == "nodes":
                    continue
                print("# " + key + " = " + value, file=f)
            for node in tree["nodes"][1:]:  # Skip the root
                print(node["index"], node["form"], node["lemma"], node["upos"],
                      node["xpos"], "_", node["head"], node["deprel"], "_", "_",
                      sep="\t", file=f)
            print(file=f)  # Separate sentences by empty line


def evaluate_las(gold, pred):
    """
    Save a Python object or dict representing a sequence of parsed sentence in back to a .conllu file.
    Args:
        gold: The gold-standard sentences or file containing them.
        pred: The parsed sentences or file containing them.
    Returns:
        LAS (labeled attachment score) averaged across all sentences.
    """
    gold = try_load_conllu(gold)
    pred = try_load_conllu(pred)
    total = correct = 0
    for gold_tree, pred_tree in zip(gold, pred):
        for gold_node, pred_node in zip(gold_tree["nodes"][1:], pred_tree["nodes"][1:]):
            total += 1
            if gold_node["head"] == pred_node["head"] and gold_node["deprel"] == pred_node["deprel"]:
                correct += 1
    return correct / total


def try_load_conllu(file_or_sentences):
    try:
        return load_conllu(file_or_sentences)
    except TypeError:
        return file_or_sentences


if __name__ == "__main__":
    import os
    dev_file_path = os.path.join("data", "ud", "da_ddt-ud-dev.conllu")
    dev_data = load_conllu(dev_file_path)
    dev_copy_file_path = os.path.join("data", "ud", "da_ddt-ud-dev_copy.conllu")
    save_to_conllu(dev_data, dev_copy_file_path)
    print(evaluate_las(dev_file_path, dev_copy_file_path))
    train_file_path = os.path.join("data", "ud", "en_ewt-ud-train.conllu")
    train_data = load_conllu(train_file_path)
    train_copy_file_path = os.path.join("data", "ud", "en_ewt-ud-train_copy.conllu")
    save_to_conllu(train_data, train_copy_file_path)
    print(evaluate_las(train_file_path, train_copy_file_path))
