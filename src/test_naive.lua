--- 
-- Commands 
--  
--  run.sh 'naive_learning_rate1' 'test_naive.lua 1' 
--  for i in $(seq 1 6); do run.sh 'naive_learning_rate1' 'test_naive.lua 1' 

require 'paths'
local naive = require 'naive'
local utils = require 'utils'

local idx= tonumber(arg[1])

local nConf = 6
if idx == nil or idx < 1 or idx > nConf then
  print('Usage: test_navie <idx>')
  print('       idx must be an integer between [%d, %d]\r\n', 1, nConf)
  return
end



local learningRates = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6}
local learningRate = learningRates[idx]
local project_dir=utils.project_dir()
local output = string.format('%s/buffer/naive_gray_inverse_learning_learningrate/%.2e', project_dir, learningRate)
paths.mkdir(output)

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
      fn_evals_txt   = string.format('%s/evals.txt', output),
      fn_evals_txt   = string.format('%s/evals.txt', output),
      fn_evals_svg   = string.format('%s/errs_vs_epoch.svg', output),
      fn_performance = string.format('%s/performance.txt', output),
      fn_model       = string.format('%s/model.dat', output),
      fn_parameters  = string.format('%s/parameters.txt', output)
}

naive.run(param)
