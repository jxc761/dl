require 'torch'
require 'nn'
require 'math'
require 'optim'
require 'gnuplot'
require 'paths'
require 'sys'

-- gup support
require 'cunn'
require 'cutorch'



local ols = require 'ols'


local naive = {}


local function save_params(filename, p)

  local file = assert( io.open(filename, 'w'))
  
  file:write(string.format('%16s=%s\r\n', 'input.dtype', p.input.dtype) )
  file:write(string.format('%16s=%s\r\n', 'input.ctype', p.input.ctype) )
  file:write(string.format('%16s=%d\r\n', 'input.res', p.input.res))

  file:write(string.format('%16s=%s\r\n', 'target.dtype', p.target.dtype) )
  file:write(string.format('%16s=%s\r\n', 'target.ctype', p.target.ctype) )
  file:write(string.format('%16s=%d\r\n', 'target.res', p.target.res))
  
  file:write(string.format('%16s=%s\r\n', 'use_gpu', p.usegpu))
  
  file:write(string.format('%16s=%d\r\n', 'nTrain', p.nTrain))
  file:write(string.format('%16s=%d\r\n', 'nTest', p.nTest))
  file:write(string.format('%16s=%d\r\n', 'nValid', p.nValid))
  file:write(string.format('%16s=%d\r\n', 'batchsz', p.batchsz))
  

  file:write(string.format('%16s=%e\r\n', 'learningRate', p.learningRate))

  if  p.fn_evals_txt then
    file:write(string.format('%16s=%s\r\n', 'fn_evals_txt', p.fn_evals_txt))
  end
  
  if p.fn_evals_svg then
    file:write(string.format('%16s=%s\r\n', 'fn_evals_svg', p.fn_evals_svg))
  end 
  
  if p.fn_perfromance then
    file:write(string.format('%16s=%s\r\n', 'fn_performance', p.fn_performance))
  end

  if p.fn_model then 
     file:write(string.format('%16s=%s\r\n', 'fn_model', p.fn_model))
 end

  if p.fn_parameters then
    file:write(string.format('%16s=%s\r\n', 'fn_params', p.fn_parameters))
  end
  
  file:close()
  
end

function naive.run(p)
  torch.manualSeed(0)
  torch.setdefaulttensortype('torch.FloatTensor')

  -- unpack parameters
  local input = p.input
  local target = p.target
  
  local nTrain = p.nTrain
  local nTest  = p.nTest
  local nValid = p.nValid
  local batchsz = p.batchsz
  
  local usegpu = p.usegpu
  local learningRate = p.learningRate
  
  local fn_evals_txt = p.fn_evals_txt
  local fn_evals_svg = p.fn_evals_svg
  local fn_performance = p.fn_performance
  local fn_model = p.fn_model
  local fn_params = p.fn_parameters
  
  -- save parameters
  if fn_params then
    save_params(fn_params, p)
  end
  
  local os = paths.uname()
  
  
  -- load data 
  -- local images = ols.LoadDataset('image',    'gray', res)
  -- local depths = ols.LoadDataset('depth', 'inverse', res)
  
  -- local X = images:view(-1, res*res)
  -- local Y = depths:view(-1, res*res)
  
  local X = ols.LoadDataset(input.dtype, input.ctype, input.res)
  local Y = ols.LoadDataset(target.dtype, target.ctype, target.res)
  X = X:view(-1, input.res*input.res)
  Y = Y:view(-1, target.res*target.res)
  if usegpu then
    X = X:cuda()
    Y = Y:cuda()
  end

  -- split data to training, validation, testing set
  local trainX = X:narrow(1, 1, nTrain)
  local trainY = Y:narrow(1, 1, nTrain)
  local validX = X:narrow(1, nTrain+1, nValid)
  local validY = Y:narrow(1, nTrain+1, nValid)  
  local testX  = X:narrow(1, nTrain+nValid+1, nTest)
  local testY  = Y:narrow(1, nTrain+nValid+1, nTest)  
  
  
  -- build model
  local md = nn.Sequential()
  local dim_x = X:size(2)
  local dim_y = Y:size(2)
  local sz = {dim_x, math.ceil(dim_x*1.2), math.ceil(dim_x*1.44), math.ceil(dim_y*1.2), dim_y }
  for i = 1, #sz-1 do
    md:add(nn.Linear(sz[i], sz[i+1]))
    md:add(nn.ReLU())
  end
  local criterion = nn.MSECriterion()
  
  -- gpu support
  if usegpu then
    md:cuda()
    criterion:cuda()
  end
  
 
  -- for debug
  -- print(md)
  -- print(criterion)
  -- print(trainX:size())
  -- print(trainY:size())
  -- print(testX:size())
  -- print(testY:size())
  
  -- open log file for recording resulting during evaluation
  local file_evals=nil
  if fn_evals_txt then
    file_evals = assert( io.open(fn_evals_txt, 'w'))
    file_evals:write(string.format('%16s%16s%16s\r\n', '#epoch', 'time(s)', 'fval') )
  end
  
  -- evaluate setting
  local evals = {}
  local durations = {}
  local epochs = {}
  local frecordevals = function (cur_epoch, cur_duration, cur_loss)
    evals[#evals+1] = cur_loss
    durations[#durations+1] = cur_duration
    epochs[#epochs+1] = cur_epoch
    
    if file_evals then
      file_evals:write( string.format('%16d%16.2f%16.4e\r\n', cur_epoch, cur_duration, cur_loss) )
      file_evals:flush()
    end
    
    if fn_evals_svg then
      -- plot the evaluate result
      if os == 'Darwin' then
        gnuplot.setgnuplotexe('/usr/local/bin/gnuplot')
      end
  
      gnuplot.setterm('svg')
      gnuplot.svgfigure(fn_evals_svg)
      gnuplot.plot(torch.Tensor(evals))
      gnuplot.plotflush() 
    end
  end    
  

  local nBatchs = math.floor(trainX:size(1) / batchsz)
  local config = {learningRate = learningRate}
  
  
  local params, gradParams = md:getParameters()  -- flatten model parameters
  params:normal() -- init prams

  sys.tic() -- start timer
  for epoch = 1, p.nIter do
  
  
    -- external variables: gradParams, md, criterion, trainX, trainY, p
    -- input : b:batch_index
    -- output: loss(batch), dl/dparams 
    local function feval(arg)
      gradParams:zero()
      local b = (epoch-1) % nBatchs 
      local index = b *batchsz + 1
      local size = batchsz
      local x = trainX:narrow(1, index, size) -- input
      local y = trainY:narrow(1, index, size) -- target

      local o = md:forward(x)               -- output of the network   
      local l = criterion:forward(o, y)     -- loss of the model
      local dl  = criterion:backward(o, y) -- d_loss/d_output
      md:backward(x, dl)                -- d_loss/d_parameters and d_loss / d_x
      return l, gradParams
    end
  
    
    optim.sgd(feval, params, config)
   
    -- evaluate on the validation dataset
    if epoch % p.evalPeriod == 0 then
      local duration = sys.toc()      
      local validl = criterion:forward(md:forward(validX), validY)
      frecordevals(epoch, duration, validl)
    end
    
  end
  
  if file_evals then
    file_evals:close()
  end
  
  
  -- evaluate peformance
  local perform = {}
  perform.duration = sys.toc()
  
  perform.valid = criterion:forward(md:forward(validX), validY)
  perform.test  = criterion:forward(md:forward(testX), testY)
  if usegpu then
    local step=1000
    local sum = 0
    local n = trainX:size(1)
    for index = 1, n, step do
      local size = step < (n-index+1) and step or (n-index+1) 
      local x = trainX:narrow(1, index, size)
      local y = trainY:narrow(1, index, size)
      local l = criterion:forward(md:forward(x), y)
      sum = sum + l * size
    end
    perform.train= sum / n;
  else
      perform.train = criterion:forward(md:forward(trainX), trainY)
  end
  
    
  -- save out peformance 
  if  fn_performance then
    local file_performance = assert( io.open(fn_performance, 'w'))
    file_performance:write( string.format('%16.4f%16.4e%16.4e%16.4e\r\n', used_time, perform.train, perform.valid, perform.test) )
    file_performance:close()
  end
  
  -- save model
  if fn_model ~= nil then
    torch.save(fn_model, {md=md, criterion=criterion}, 'binary')
  end
  
  return {md=md, criterion=criterion, perform=perform, process={evals=evals, epochs=epochs, durations=durations}}
end


return naive
