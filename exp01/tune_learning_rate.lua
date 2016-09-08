require 'torch'
require 'os'
require 'paths'

local utils = require 'utils'

local function get_parameter(learningrate, output)

    local param = {      
      input  = {dtype = 'image',  ctype='gray', res=16}, 
      target = {dtype= 'depth', ctype='inverse', res=16},
      nTrain = 400 * 40 * 30,
      nValid = 50 * 40 * 30,
      nTest  = 50 * 40 * 30,
      batchsz = 128,
      
      learningRate = learningrate,      
      evalPeriod = 100,
      nIter = 10000,
      usegpu=true,
      
      fn_evals_txt   = string.format('%s/proces_%e.txt', output, learningrate),
      fn_evals_svg   = string.format('%s/proces_%e.svg', output, learningrate), --string.format('%s/errs_vs_epoch.svg', output),
      fn_performance = nil, --string.format('%s/performance.txt', output),
      fn_model       = nil, --string.format('%s/model.dat', output),
      fn_parameters  = nil  --string.format('%s/parameters.txt', output)
    }
    
    return param
end

local function save_results(learningrates, results, output)
  local n = #learningrates
  
  
  -- save the learning rates out
  local fn_learningrates = string.format('%s/learningrates.txt', output)
  local flearningrates = assert( io.open(fn_learningrates, 'w'))
  for i=1,n do
    flearningrates:write(string.format('%f\r\n', learningrates[i]))
  end
  flearningrates:close()
  
  -- save the performance out
  local fn_perform = string.format('%s/performances.txt', output)
  local fperform = assert( io.open(fn_perform, 'w'))
  fperform:write('learning_rate\tduration\ttraining\tvalidation\ttesting\r\n') 
  for i=1,n do
    local pi=results[i].perform
    fperform:write(string.format('%f\t%.2f\t%f\t%f\t%f\r\n', learningrates[i], pi.duration, pi.train, pi.valid, pi.test))
  end
  fperform:close()
  
  -- local project_dir = utils.project_dir()
  -- local plot_script = string.format('%s/exp01/plot_results.gpl', project_dir)
  -- local plot_result = string.format('%s/performances.eps', output)
  -- local plot_cmd = string.format('gnuplot -e "input=%s; output=%s;" %s', output, plot_result, plot_script)
  -- print(plot_cmd)
  -- os.execute(plot_cmd)
  
end


local function tune_learningrate(min, max, n, method)
  local learningrates
  if method == 'log' then
    learningrates = torch.logspace(math.log10(min), math.log10(max), n)
  else
    learningrates = torch.linespace(min, max, n)
  end
  
  local output = string.format('%s/buffer/tune_lr_%f_%f_%d_%s', utils.project_dir(), min, max, n,  method)
  paths.mkdir(output)
  
  local results = {}
  for idx = 1, #learningrates do
    local param = get_parameter(learningrates[idx], output)
    results[idx] = naive.run(param)
  end
  
  save_results(learningrates, results, output)
  
end

local varargs = {...}

local min = varargs[1] and tonumber(varargs[1]) or 1e-6
local max = varargs[2] and tonumber(varargs[2]) or 1e-1
local n   = varargs[3] and tonumber(varargs[3]) or 6
local method = varargs[4] and varargs[4] or 'log'

print('tuning learning rate......')
print('min=' .. min)
print('max=' .. max)
print('n='.. n)
print('method=' .. method)

tune_learningrate(min, max, n, method)

print('finish')
