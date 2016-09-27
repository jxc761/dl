set -o nounset
set -o errexit

remote_project="jxc761@hpc1.case.edu:~/dl"
local_project="/Users/Jing/Dropbox/dev/dl"
cur_dir=$1

rsync --checksum --times --verbose --recursive ${remote_project}/${cur_dir} ${local_project}
