require 'paths'

local utils = require 'utils'
local naive = require 'naive'

local function run()

    local project_dir=utils.project_dir()
    local output = string.format('%s/buffer/exp01/run', project_dir)
    paths.mkdir(output)
    
    local param = {      
      input  = {dtype = 'image',  ctype='gray', res=16}, 
      target = {dtype= 'depth', ctype='inverse', res=16},
      nTrain = 400 * 40 * 30,
      nValid = 50 * 40 * 30,
      nTest  = 50 * 40 * 30,
      batchsz = 128,
      learningRate = 0.14,
      evalPeriod = 100,
      nIter = 100000,
      usegpu= true,
      fn_evals_txt   = string.format('%s/process.txt', output),
      fn_evals_svg   = nil, string.format('%s/process.svg', output),
      fn_performance = string.format('%s/performance.txt', output),
      fn_model       = nil, --string.format('%s/model.dat', output),
      fn_parameters  = string.format('%s/parameters.txt', output)
    } 
    
    naive.run(param)
end


run()

