import json

def get_negated_data(file):
    data = []
    for line in open(file):
        data.append(json.loads(line))
    negated = []
    for line in data:
        if line["negations"] > 0:
            negated.append(line)
    return negated

def convert_to_tagged(data):
    tagged = {}
    for sentence in data:
        x, y = [], []
        for node in sentence["nodes"]:
            neg = node["negation"][0]
            x.append(node["form"])
            if "scope" in neg.keys():
                y.append("scope")
            elif "cue" in neg.keys():
                y.append("cue")
            else:
                y.append("O")

        # make sentence a string so we can get the set in a second
        x = " ".join(x)
        if x not in tagged:
            tagged[x] = []

        tagged[x].append(y)
    return tagged

def combine_scopes(scopes):
    final = scopes[0]
    for scope in scopes[1:]:
        for i, tag in enumerate(scope):
            if tag == "scope" and final[i] == "O":
                final[i] = "scope"
            if tag == "cue" and final[i] in ["scope", "O"]:
                final[i] = "cue"
    return final

def bio_tag(scope):
    tags = []
    cueb = False
    scopeb = False
    
    for tag in scope:
        if tag == "cue":
            scopeb = False
            if cueb == False:
                tags.append("B_cue")
                cueb = True
            elif cueb == True:
                tags.append("I_cue")
        elif tag == "scope":
            cueb = False
            if scopeb == False:
                tags.append("B_scope")
                scopeb = True
            else:
                tags.append("I_scope")
        else:
            tags.append("O")
            cueb = False
            scopeb = False

    return tags

def read_data(file):
    negated = get_negated_data(file)
    tagged = convert_to_tagged(negated)
    combined = []
    for sent, tags in tagged.items():
        if len(tags) > 1:
            tags = [combine_scopes(tags)]
        combined.append((sent.split(), bio_tag(tags[0])))
    return combined
    
