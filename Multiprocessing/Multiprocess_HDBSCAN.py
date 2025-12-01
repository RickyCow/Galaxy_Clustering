import utils
import itertools
import logging
import psutil
import glob


def main():
	files = glob.glob("map_outputs/umap_embeddings_*.csv")

    for f in files:
        d = utils.pd.read_csv(f'{f}')
        results = utils.opt(d, trials = 100)
        data_copy = data.copy()
        data_copy['labels'] = results['labels']
        utils.concat[data_copy,d,axis==1].to_csv(f'umap_outputs/results_{i}', index=False)
	    
main()
