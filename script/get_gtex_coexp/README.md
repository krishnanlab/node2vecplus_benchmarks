Files
* `tissues.txt` - all gtex coexpression dab files
* `annotated_tissues.txt` - selected files that have corresponding GO tasks (from OhmNet)

Scripts
* `run.sh` - main run script
* `download.sb` - raw data download job
* `convert_dat.sb` - dab to dat (edge list) conversion jobs
* `convert_npz.sb` - dat to npz (adjacency matrix as dense numpy array with node IDs info) conversion jobs
