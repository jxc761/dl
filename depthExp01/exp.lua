local DataSource = require 'Data'


local function run(opts)
  local data    = DataSource.DataSource(opts.data)
  local model   = Model(opts.model)
  
  local monitor = Monitor(opts.monitor, data, model)

  -- train model with automatically 
  local path_to_output = 
  local batchsz = opts.batchsz
  local nIter   = opts.nIter or math.floor( data.nTrain/opts.batchsz)
  local sampler = torch.randperm(1, nIter*batchsz):fmod(data.nTrain)

  for i = 1, n do
  	model:reset()
    local monitor = Monitor({fn_txt=string.format('%s/%e/process.txt', opts.output, lr), evalPeriod=100})
    local opt = {batchsz=batchsz, nIter = nIter, sampler=sampler, sgd}
    train(model, data, monitor, evaluator, opt)

    evaluate(model, data, opts)
  end

end






local function auto_train(model, data, opts)

end


lacal varargs = {...}
local fconf = varargs[1]
local index = tonumber(varargs[2])

local configurator = require(fconf)
local opts=configurator.conf(index)

run(opts)
 
  