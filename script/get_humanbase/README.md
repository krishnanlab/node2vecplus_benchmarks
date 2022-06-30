Files
* `tissues.txt` - all tissues specific networks available on HumanBase
* `annotated_tissues.txt` - selected tissues with corresponding GO tasks (from OhmNet)
* `valid_annotated_tissues.txt` - same as the above but only for those that have sufficient (n>=5) number of positives per split

Scripts
* `run.sh` - main run script
* `download.sb` - raw data download job
* `convert_npz.sb` - convert edge lists to dense array npz jobs
