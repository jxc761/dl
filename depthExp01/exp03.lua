require'torch'
require'build_ds'
require'build_md'
require'build_tr'
require'Examples'
require'paths'
require'run'
require'utils'



torch.manualSeed(0)
torch.setdefaulttensortype('torch.FloatTensor')

local data        = build_ds()
local model       = build_md()
local examples    = Examples(data)
local trainOpts   = getTrainOpts({min=1, max=10, n=4, method='linear'}, data.nTrain)
local root_output = string.format('%s/depthExp01/d01_m01_t03', utils.buffer_dir())

paths.mkdir(root_output)

print(data)
print(model)
print(examples)
-- print(trainOpts)
print(root_output)

run(model, data, examples, trainOpts, root_output)
