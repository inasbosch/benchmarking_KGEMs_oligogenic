from config.embedding_source_paths import RVIS_FILE, HGNC_ENSEMBL_FILE

from numpy import ndarray, asarray
import pandas as pd
from typing import Union, List, Tuple

class Dataset():
    def __init__(self, bock_file: str, edge2vec_file: str):
        self.bock_file = bock_file
        self.edge2vec_file = edge2vec_file
        self.entity_id_to_label = {}
        self.entity_label_to_id = {}
        self.relation_label_to_id = {}

    def append_id_to_label(self, label: str, entity: bool):
        if entity:
            mapping_dict = self.entity_label_to_id
        else:
            mapping_dict = self.relation_label_to_id

        if label not in mapping_dict:
            mapping_dict[label] = len(mapping_dict)

    def write_txt_file(self):
        with open(self.bock_file, "r") as f:
            with open(self.edge2vec_file, "w") as g:
                for i, row in enumerate(f):
                    split_row = row.split("\t")
                    node_label1 = split_row[0]
                    self.append_id_to_label(node_label1, True)
                    if self.entity_label_to_id[node_label1] not in self.entity_id_to_label:
                        self.entity_id_to_label[self.entity_label_to_id[node_label1]] = node_label1

                    node_label2 = split_row[2]
                    self.append_id_to_label(node_label2, True)
                    if self.entity_label_to_id[node_label2] not in self.entity_id_to_label:
                        self.entity_id_to_label[self.entity_label_to_id[node_label2]] = node_label2

                    edge_type = split_row[1]
                    self.append_id_to_label(edge_type, False)

                    g.write(str(self.entity_label_to_id[node_label1]) + " " + str(self.entity_label_to_id[node_label2]) + " "
                            + str(self.relation_label_to_id[edge_type]) + " " + str(i) + "\n")

def rvis_mapping():
    mapping_dict = {}

    hgnc_to_ensembl_dict, _ = hgnc_ensembl_mapper()

    df = pd.read_excel(RVIS_FILE, header=None)
    f = df.values.tolist()[1:]

    for row in f:
        gene_name = row[0]
        ensembl_id_set = hgnc_to_ensembl_dict[gene_name]
        rvis_value = float(row[1])

        for id in ensembl_id_set:
            mapping_dict[id] = rvis_value

    return mapping_dict


def sort_genes(array: Union[ndarray,List[Tuple[str,str]]]):
    rvis_dict = rvis_mapping()

    new_array = []
    for row in array:
        genes = [row[0],row[1]]
        if genes[0] in rvis_dict and genes[1] in rvis_dict:
            sorted_genes = sorted(genes,key=lambda x: rvis_dict[x]) + list(row[2:])
        else:
            if genes[0] in rvis_dict:
                sorted_genes = genes + list(row[2:])
            else:
                sorted_genes = [genes[1],genes[0]] + list(row[2:])

        new_array.append(sorted_genes)

    return asarray(new_array)

def hgnc_ensembl_mapper():
    hgnc_to_ensembl_dict = {}
    ensembl_to_hgnc_dict = {}
    with open(HGNC_ENSEMBL_FILE, "r") as f:
        for row in f:
            split_row = row.split("\t")
            if split_row[3] == "Approved":
                hgnc_id = split_row[1]
                ensembl_id = split_row[-1][:-1]

                if hgnc_id not in hgnc_to_ensembl_dict:
                    hgnc_to_ensembl_dict[hgnc_id] = [ensembl_id]
                else:
                    hgnc_to_ensembl_dict[hgnc_id].append(ensembl_id)

                if ensembl_id not in ensembl_to_hgnc_dict:
                    ensembl_to_hgnc_dict[ensembl_id] = [hgnc_id]
                else:
                    ensembl_to_hgnc_dict[ensembl_id].append(hgnc_id)

    return hgnc_to_ensembl_dict, ensembl_to_hgnc_dict

if __name__ == '__main__':
    hgnc_ensembl_mapper()
