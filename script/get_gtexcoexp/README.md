Files
* `tissues.txt` - all gtex coexpression dab files
* `annotated_tissues.txt` - selected files that have corresponding GO tasks (from OhmNet)
* `valid_annotated_tissues.txt` - same as the above but only for those that have sufficient (n>=5) number of positives per split
* `valid_annotated_tissues_converted.txt` - converted tissue names (using HumanBase naming) aligned with `valid_annotated_tissue.txt`

Job scripts
* `run.sh` - main run script
* `download.sb` - raw data download job
* `convert_dat.sb` - dab to dat (edge list) conversion jobs
* `convert_npz.sb` - dat to npz (adjacency matrix as dense numpy array with node IDs info) conversion jobs
* `sparsify_networks.sb` - run `sparsify_networks.py` job

Scripts
* `sparsify_networks.py` - compute the optimal cut threshold and use it to construct and save the sparsified GTEx coexpression networks.
    * 2020-07-20: 0.7368
    * 2022-07-22: 2.2105 (added GTExCoExp-global; optim cut based on 98% genes preserved in the largest connected component)
