Files
* `tissues.txt` - all gtex coexpression dab files
* `annotated_tissues.txt` - selected files that have corresponding GO tasks (from OhmNet)
* `valid_annotated_tissues.txt` - same as the above but only for those that have sufficient (n>=5) number of positives per split
* `valid_annotated_tissues_converted.txt` - converted tissue names (using HumanBase naming) aligned with `valid_annotated_tissue.txt`

Scripts
* `run.sh` - main run script
* `download.sb` - raw data download job
* `convert_dat.sb` - dab to dat (edge list) conversion jobs
* `convert_npz.sb` - dat to npz (adjacency matrix as dense numpy array with node IDs info) conversion jobs
