require'torch'
require'build_ds'
require'build_md'
require'build_tr'
require'Examples'
require'paths'
require'run'

local utils = require'utils'



------------------------------------------------------------------------------------------
-- parse command-line options
--
local min=tonumber(arg[1])
local max=tonumber(arg[2])
local n=tonumber(arg[3])
local method=arg[4]

print('min='..min)
print('max='..max)
print('n='..n)
print('method='..method)


torch.manualSeed(0)
torch.setdefaulttensortype('torch.FloatTensor')

local data        = build_ds()
local model       = build_md()
local examples    = Examples(data)
local trainOpts   = getTrainOpts({min=min, max=max, n=n, method=method}, data.nTrain)
local root_output = string.format('%s/depth-step-by-step/%.2e_%.2e_%d_%s', utils.buffer_dir(), min, max, n, method)
paths.mkdir(root_output)

print(data)
print(model)
print(examples)
print(trainOpts)
print(root_output)



run(model, data, examples, trainOpts, root_output)
