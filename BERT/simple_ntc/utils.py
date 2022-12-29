# def read_text(fn):
#     with open(fn, 'r') as f:
#         lines = f.readlines()
#
#         labels, texts = [], []
#         for line in lines:
#             if line.strip() != '':
#                 # The file should have tab delimited two columns.
#                 # First column indicates label field,
#                 # and second column indicates text field.
#                 label, text = line.strip().split('\t') # Two columns separated by tab.
#                 labels += [label]
#                 texts += [text]
#
#     return labels, texts

def read_text(fn):
    import pandas as pd
    data = pd.read_csv(fn, header=None, sep='\t')
    # labeled_data = []
    # for row in data.iterrows():
    #     # l1, l2 = row[1][-1:][2]
    #     # labeled_data.append([row[1][0], l1])
    #     # labeled_data.append([row[1][1], l2])
    #     s, l = row[1].sentence, row[1].label
    #
    # # data_labeled = pd.DataFrame(labeled_data, columns=["sentence", "label"])
    # data_labeled = pd.DataFrame(labeled_data, columns=["sentence", "label"])
    sentence, label = data[0].to_list(), data[1].to_list()
    return label, sentence



def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm
