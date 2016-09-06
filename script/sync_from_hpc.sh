set -o nounset
set -o errexit

remote_project="jxc761@hpc1.case.edu:~/depth"
local_project="/Users/Jing/Dropbox/dev/depth"

rsync --checksum --times --verbose --recursive ${remote_project}/buffer ${local_project}
rsync --checksum --times --verbose --recursive ${remote_project}/log ${local_project}