from pathlib import Path
import tarfile
import gzip
import shutil
from collections import defaultdict
import random
import logging
import multiprocessing


import requests
import tqdm
from .moco_dataset import MoCoKGDataset

logging.basicConfig(level=logging.INFO)


class WikiData5M(MoCoKGDataset):
    name: str = "WikiData5M"
    path: Path = Path(__file__).parents[0] / "WikiData5M"
    mode: str

    def __init__(self, mode:str, sigma:int = 8, tranductive:bool = True):
        """
        Initialize the WikiData5M dataset.
        :param mode: The mode of the dataset. Either "train", "test" or "valid".
        :param sigma: The number of neighbors to consider for each entity.
        :param tranductive: if the tranductive dataset will be downloaded (else inductive).   
        """
        self.path = Path(self.path)

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

        self.name = self.name
        self.mode = mode
        self.sigma = sigma
        self.tranductive = tranductive

        if len(list(self.path.rglob("*.txt"))) < 6:
            self.download()
        # Tuples containing (head id, relation id, tail id)
        self.links = []
        # Mapping from entity id to a list of indices of the links
        self.neighborhoods = defaultdict(list)
        knowledge_graph_file = self.path / f"wikidata5m_{'transductive' if tranductive else 'inductive'}_{self.mode}.txt"
        
        with knowledge_graph_file.open() as f:
            link_tqdm = tqdm.tqdm(f.readlines())
            link_tqdm.set_description(f"Processing {mode.upper()} Link")
            for i, line in enumerate(link_tqdm):
                head_id, relation_id, tail_id = line.strip().split("\t")
                self.links.append((head_id, relation_id, tail_id))
                self.neighborhoods[head_id].append(i)
                self.neighborhoods[tail_id].append(i)
        entity_file = self.path / f"wikidata5m_entity.txt"
        relations_file = self.path / f"wikidata5m_relation.txt"


        # Mapping from entity id to a set of aliases
        self.entity_aliases = defaultdict(list)
        # Mapping from entity to it's id
        self.entity_to_id = defaultdict(str)
        # Mapping from relation id to a set of aliases
        self.relation_aliases = defaultdict(list)
        # Mapping from relation to it's id
        self.relation_to_id = defaultdict(str)
        for curr_file, curr_dict, to_id in tqdm.tqdm(zip([entity_file, relations_file], [self.entity_aliases, self.relation_aliases], [self.entity_to_id, self.relation_to_id])):
            with curr_file.open() as f:
                aliases_tqdm = tqdm.tqdm(f.readlines())

                aliases_tqdm.set_description(f"Processing {mode.upper()} Alias from {curr_file.name}")
                for i, line in enumerate(aliases_tqdm):
                    curr_id = line.strip().split("\t")[0]
                    aliases = line.strip().split("\t")[1:]
                    curr_dict[curr_id].extend(aliases)
                    for alias in aliases:
                        to_id[alias] = curr_id

        # Mapping from ids to descriptions, reading corpus
        self.id_to_description = defaultdict(str)
        corpus_file = self.path / f"wikidata5m_text.txt"
        with corpus_file.open() as f:
            corpus_tqdm = tqdm.tqdm(f.readlines())

            corpus_tqdm.set_description(f"Processing {mode.upper()} Corpus from {corpus_file.name}")
            for i, line in enumerate(corpus_tqdm):
                curr_id = line.strip().split("\t")[0]
                description = line.strip().split("\t")[1]
                self.id_to_description[curr_id] += description
                

        # Unique entity_ids
        self.unique_entity_ids = set(self.entity_aliases.keys())

    def find_neighbors(self, entity):
        """
        Find the neighbors of an entity in the knowledge graph.
        :param entity: The entity to find the neighbors of.
        :return: A list of neighbors of the entity.
        """
        if entity not in self.entity_to_id:
            raise ValueError(f"Entity {entity} not found in the knowledge graph.")
        entity_id = self.entity_to_id[entity]

        if entity_id not in self.neighborhoods:
            raise ValueError(f"Entity {entity} with id {entity_id} not found in the neighborhoods.")
        

        neighbors = [
            self.links[i]
            for i
            in self.neighborhoods[entity_id]
        ]
        #nomalize size of neighborhood
        if len(neighbors) > self.sigma:
            # Sample without replacement so there are not duplicates
            neighbors = random.sample(neighbors, min(len(neighbors), self.sigma))
            
        return [
            (self.entity_aliases[head_id], self.relation_aliases[relation_id], self.entity_aliases[tail_id])
            for head_id, relation_id, tail_id in neighbors
        ]

    def __len__(self):
        return len(self.links)

    def __getitem__(self, index):
        head_id, relation_id, tail_id = self.links[index]
        head = self.entity_aliases[head_id]
        relation = self.relation_aliases[relation_id]
        tail = self.entity_aliases[tail_id]
        return head, relation, tail
    

    def clean(self):
        """
        Clean the local content Wikidata 5M dataset. 
        """
        shutil.rmtree(self.path)

    def download(self):
        """
        Download the Wikidata 5M dataset. Particularly the url
        """
        knowledge_graph_url = \
            f"https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_{'transductive' if self.tranductive else 'inductive'}.tar.gz?dl=1"
        corpus_url = "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1"
        entity_relation_aliases_url = "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1"
        urls = [knowledge_graph_url, corpus_url, entity_relation_aliases_url]
        knowledge_graph_filename = f"wikidata5m_{'transductive' if self.tranductive else 'inductive'}.tar.gz"
        corpus_filename = "wikidata5m_text.txt.gz"
        entity_relation_aliases_filename = "wikidata5m_alias.tar.gz"
        url_filenames = [
            knowledge_graph_filename, 
            corpus_filename, 
            entity_relation_aliases_filename
        ]
        # Download zipped files
        for url, url_filename in zip(urls, url_filenames):
            logging.info(f"Downloading {url}...")
            file_path_url = self.path / url_filename
            if not file_path_url.exists():
                r = requests.get(url, allow_redirects=True)
                open(file_path_url, 'wb').write(r.content)
            else:
                logging.info(f"{file_path_url} already exists.")
        
        # Extract zipped files
        for url_filename in [knowledge_graph_filename, entity_relation_aliases_filename]:
            logging.info(f"Extracting {url_filename}...")
            with tarfile.open(self.path / url_filename, "r:gz") as tar:
                tar.extractall(path=self.path)
        # Extract corpus
        with gzip.open(self.path / corpus_filename, 'rb') as f_in:
            with open(self.path / corpus_filename.replace(".gz", ""), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # Clean up compressed files
        for url_filename in url_filenames:
            (self.path / url_filename).unlink()
        logging.info("Done!")
