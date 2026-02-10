import utils
import itertools
import logging
import psutil
import sys
import glob


def main():
	galaxy = sys.argv[1]

	utils.os.makedirs("logs", exist_ok=True)
	logging.basicConfig(
	    filename="logs/umap_hdbscan_runs.log",
	    level=logging.INFO,
	    format="%(asctime)s [%(levelname)s] %(message)s"
	)

	def log_resource(stage, run_id=None):
	    """Log current RAM usage (GB)"""
	    process = psutil.Process(utils.os.getpid())
	    mem_gb = process.memory_info().rss / 1e9
	    logging.info(f"Run {run_id} | {stage} | RAM: {mem_gb:.2f} GB")
	    print(f"[Run {run_id}] {stage}: RAM {mem_gb:.2f} GB")
	    return mem_gb
	    
	data = utils.pd.read_csv(f'{galaxy}')
	interest = ['age','feh','alpha','V','UW']
	data = data[['age','feh','alpha','V','UW']]

	# It is good practice to standardize the data
	# There are many different types of standardization
	X = utils.StandardScaler().fit_transform(data).astype(utils.np.float32)

	hist_kwargs = {
	    'histtype' : 'step',
	    'bins'     : 100,
	    'density'  : True
	}


	for i in range(data.shape[1]):
	    utils.plt.hist(
		data.values[:, i],
		label = 'Before standardization',
		**hist_kwargs
	    )

	    utils.plt.hist(
		X[:, i],
		label = 'After standardization',
		**hist_kwargs
	    )

	    utils.plt.xlabel(f'Variable {i}')
	    utils.plt.ylabel('Density')
	    utils.plt.legend(loc='best')

	    utils.plt.show()
	    
	umap_param_grid = {
	    "n_neighbors": utils.np.linspace(10,55,num=10),
	    "min_dist": utils.np.linspace(0.0,0.9,num=10),
	    "n_epochs": [1000],
	    "verbose": [True],
	    "set_op_mix_ratio": utils.np.linspace(0.0,0.9,num=10),
	    "local_connectivity": utils.np.linspace(1,5,num=10)
	}

	combinations = list(itertools.product(
	    umap_param_grid["n_neighbors"],
	    umap_param_grid["min_dist"],
	    umap_param_grid["n_epochs"],
	    umap_param_grid["verbose"],
	    umap_param_grid["set_op_mix_ratio"],
	    umap_param_grid["local_connectivity"]
	))
	logging.info(f"Total combinations: {len(combinations)}")

	def run_umap_on_core(args):
		"""Run a single UMAP embedding pinned to a specific CPU core."""
		idx, (n_neighbors,min_dist,n_epochs,verbose,set_op_mix_ratio,local_connectivity) = args
		core_id = idx % utils.os.cpu_count()  # assign one core per process
		run_id = f"UMAP_{idx:02d}_core{core_id}_n{n_neighbors}_d{min_dist}_e{n_epochs}"
		
		try:
			# Pin process to one CPU core
			utils.os.sched_setaffinity(0, {core_id})
			print(f"\n[{run_id}] Starting on CPU core {core_id}")
			logging.info(f"[{run_id}] Starting on core {core_id}")

			log_resource("Before UMAP", run_id)
			utils.wrapper(idx, X, n_neighbors, min_dist, n_epochs, verbose, set_op_mix_ratio, local_connectivity)
			log_resource("After UMAP", run_id)

			logging.info(f"[{run_id}] Completed successfully")
			print(f"[{run_id}] Completed")
			
		except Exception as e:
			logging.error(f"[{run_id}] Failed: {e}")
			print(f"[{run_id}] Failed: {e}")
		
	print("Running UMAPs in parallel (1 process per core)")
	available_cores = utils.os.cpu_count()
	processes = min(len(combinations), available_cores)
	logging.info(f"Using {processes} parallel processes (one per core).")

	with utils.mp.Pool(processes=processes) as pool:
	    pool.map(run_umap_on_core, enumerate(combinations))

	print("All UMAP embeddings generated.")
	logging.info("All UMAP embeddings completed.")
    files = glob.glob("map_outputs/umap_embeddings_*.csv")

    for f in files:
        d = utils.pd.read_csv(f'{f}')
        results = utils.opt(d, trials = 100)
        data_copy = data.copy()
        data_copy['labels'] = results['labels']
        utils.concat[data_copy,d,axis==1].to_csv(f'umap_outputs/results_{i}', index=False)

main()
