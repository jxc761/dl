--- 
-- Commands 
--  
--  run.sh 'naive_learning_rate1' 'test_naive.lua 1' 
--  for i in $(seq 1 6); do run.sh 'naive_learning_rate1' 'test_naive.lua 1' 

require 'torch'
require 'paths'
require 'math'


local naive = require 'naive'
local utils = require 'utils'


torch.manualSeed(0)

local function test_using_gpu(varargs)

    local usegpu=tonumber(varargs[1]) ~= 0
    local project_dir=utils.project_dir()
    local output = string.format('%s/buffer/naive_gray_inverse_gpu/usegpu_%s', project_dir, usegpu)

    paths.mkdir(output)
    
    local param = {      
      input  = {dtype = 'image',  ctype='gray', res=16}, 
      target = {dtype= 'depth', ctype='inverse', res=16},
      nTrain = 400 * 40 * 30,
      nValid = 50 * 40 * 30,
      nTest  = 50 * 40 * 30,
      batchsz = 128,
      learningRate = 0.07,
      evalPeriod = 100,
      nIter = 10000,
      usegpu=usegpu,
      fn_evals_txt   = string.format('%s/evals.txt', output),
      fn_evals_svg   = string.format('%s/errs_vs_epoch.svg', output),
      fn_performance = string.format('%s/performance.txt', output),
      fn_model       = nil, --string.format('%s/model.dat', output),
      fn_parameters  = string.format('%s/parameters.txt', output)
    } 
    
    naive.run(param)
     
end



local function test_learning_rate(learningRate)

    local project_dir=utils.project_dir()
    local output = string.format('%s/buffer/naive_gray_inverse_learningrate/%.6f', project_dir, learningRate)
    paths.mkdir(output)
    -- utils.mkdir(output) 

    local param = {      
      input  = {dtype = 'image',  ctype='gray', res=16}, 
      target = {dtype= 'depth', ctype='inverse', res=16},
      nTrain = 400 * 40 * 30,
      nValid = 50 * 40 * 30,
      nTest  = 50 * 40 * 30,
      batchsz = 128,
      learningRate = learningRate,
      evalPeriod = 100,
      nIter = 10000,
      usegpu=false,
      fn_evals_txt   = string.format('%s/evals.txt', output),
      fn_evals_svg   = string.format('%s/errs_vs_epoch.svg', output),
      fn_performance = string.format('%s/performance.txt', output),
      fn_model       = nil, --string.format('%s/model.dat', output),
      fn_parameters  = string.format('%s/parameters.txt', output)
    }

    naive.run(param)
end


local function coarse_tuning_learningrate(varargs)

    -- local varargs = {...}
    local idx = tonumber(varargs[1])
    local learningRates = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6}
    test_learning_rate(learningRates[idx])

end

local function fine_tuning_learningrate(varargs)

    -- local varargs = {...}
    local minLR = tonumber(varargs[1])
    local maxLR = tonumber(varargs[2])
    local n = tonumber(varargs[3])
    local i = tonumber(varargs[4])

    print(string.format('minLR= %.4e, maxLR=%.4e, n=%d, i=%d\r\n', minLR, maxLR, n, i) )
    local fineLearningRates = torch.logspace(math.log10(minLR), math.log10(maxLR), n)

    test_learning_rate(fineLearningRates[i])

end


local task= table.remove(arg, 1)
local funcs = {
    fine_tuning = fine_tuning_learningrate,
    coarse_tuning = coarse_tuning_learningrate,
    test_gpu = test_using_gpu
}
funcs[task](arg)

