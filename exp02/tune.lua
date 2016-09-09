
require 'paths'

local utils = require 'utils'
local naive2 = require 'naive2'

local function save_results(learningrates, results, output)
  local n = learningrates:nElement()
  
  
  -- save the learning rates out
  local fn_learningrates = string.format('%s/learningrates.txt', output)
  local flearningrates = assert( io.open(fn_learningrates, 'w'))
  for i=1,n do
    flearningrates:write(string.format('%e\r\n', learningrates[i]))
  end
  flearningrates:close()
  
  -- save the performance out
  local fn_perform = string.format('%s/performances.txt', output)
  local fperform = assert( io.open(fn_perform, 'w'))
  fperform:write('learning_rate\tduration\ttraining\tvalidation\ttesting\r\n') 
  for i=1,n do
    local pi=results[i].perform
    fperform:write(string.format('%e\t%e\t%e\t%e\t%e\r\n', learningrates[i], pi.duration, pi.train, pi.valid, pi.test))
  end
  fperform:close()
end


local function update_learning_rate(param, r, output) 
  param.learningRate   = r
  param.fn_evals_txt   = string.format('%s/process_%e.txt', output, r)
  param.fn_evals_img   = nil
  param.fn_performance = nil
  param.fn_model       = nil
  param.parameters     = nil
  
  return param
end


local function tune_learningrate(param, output, min, max, n, method)
  local learningrates
  if method == 'log' then
    learningrates = torch.logspace(math.log10(min), math.log10(max), n)
  else
    learningrates = torch.linespace(min, max, n)
  end
  
  local output = string.format('%s/%f_%f_%d_%s', output, min, max, n,  method)
  paths.mkdir(output)
  
  local results = {}
  for idx = 1, n do
    local param = update_learning_rate(param, learningrates[idx], output)
    results[idx] = naive2.run(param)
  end
  
  save_results(learningrates, results, output)
  
end




local function test(dataset, struct, nhidden, alpha, min, max, n, method)

    local output = string.format('%s/buffer/exp02/%s/%s_%d_%.2f', utils.project_dir(), dataset, struct, nhidden, alpha)

    local param = {      
      input  = {dtype = 'image',  ctype='gray', res=16}, 
      target = {dtype= 'depth', ctype='inverse', res=16},
      
      nTrain = dataset=='small' and 200 * 40 * 30 or 400 * 40 *30,
      nValid = dataset=='small' and 20 * 40 * 30 or 50 * 40 *40,
      nTest  = dataset=='small' and 20 * 40 * 30 or 50 * 40 *40,
      
      struct = struct,
      dim = 16 * 16,
      nhidden = nhidden,
      alpha = alpha,
      
      batchsz = 128,
      evalPeriod = 100,
      nIter = 10000
    } 
    
    tune_learningrate(param, output, min, max, n, method)
end



dataset=arg[1];
struct=arg[2]
nhidden=tonumber(arg[3])
alpha=tonumber(arg[4])
min=tonumber(arg[5])
max=tonumber(arg[6])
n=tonumber(arg[7]) 
method=arg[8]

print('dataset=' .. dataset)
print('struct=' .. struct)
print('nhidden=' .. nhidden)
print('alpha=' .. alpha)
print('min=' .. min)
print('max=' .. max)
print('n=' .. n)
print('method=' .. method)

test(dataset, struct, nhidden, alpha, min, max, n, method)
