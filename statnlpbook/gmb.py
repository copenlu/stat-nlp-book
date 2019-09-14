import csv

def load_gmb_dataset(file="../data/gmb/GMB_dataset_utf8.txt"):
    sents_tok, tokens = [], []
    sents_pos, pos = [], []
    sents_span, span = [], []
    current = '1.0'
    with open(file, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row['Sentence #'] != current:
                sents_tok.append(tokens)
                sents_pos.append(pos)
                sents_span.append(span)
                tokens, pos, span = [], [], []
                current = row['Sentence #']
            if not row['Word']:
                continue
            tokens.append(row['Word'])
            pos.append(row['POS'])
            span.append(row['Tag'])
        sents_tok.append(tokens)
        sents_pos.append(pos)
        sents_span.append(span)

    return sents_tok, sents_pos, sents_span
