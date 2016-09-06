require 'torch'
require 'nn'
require 'math'
require 'optim'
require 'gnuplot'
require 'paths'
require 'sys'

local ols = require 'ols'


local naive = {}

function naive.run(p)
  torch.setdefaulttensortype('torch.FloatTensor')

  -- unpack parameters
  local input = p.input
  local target = p.target
  
  local nTrain = p.nTrain
  local nTest  = p.nTest
  local nValid = p.nValid
  local batchsz = p.batchsz
  
  local learningRate = p.learningRate
  
  local fn_evals_txt = p.fn_evals_txt
  local fn_evals_svg = p.fn_evals_svg
  local fn_performance = p.fn_performance
  local fn_model = p.fn_model
  local fn_params = p.fn_parameters
  
  -- save parameters
  torch.save(fn_params, p, 'ascii')
  
  
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
  
  
  
 
  -- for debug
  -- print(md)
  -- print(criterion)
  -- print(trainX:size())
  -- print(trainY:size())
  -- print(testX:size())
  -- print(testY:size())
  
  -- open log file
  local file_evals = assert( io.open(fn_evals_txt, 'w'))
  file_evals:write(string.format('%16s%16s%16s\r\n', '#epoch', 'time(s)', 'fval') )
  
  
  -- train model
  local evals = {}
  local nBatchs = math.floor(trainX:size(1) / batchsz)
  local config = {learningRate = learningRate}
  sys.tic() -- start timer
  
  local params, gradParams = md:getParameters()  -- flatten model parameters
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
    
    -- evaluate 
    if epoch % p.evalPeriod == 0 then
      local duration = sys.toc()
      -- evaluate on the testing dataset
      local o = md:forward(validX)
      local l = criterion:forward(o, validY)
      evals[#evals+1] = l
      
      file_evals:write( string.format('%16d%16.2f%16.4e\r\n', epoch, duration, l) )
      
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
  
  file_evals:close()
  
  
  -- evaluate peformance
  local performance = {}
  performance.test  = criterion:forward(md:forward(testX), testY)
  performance.train = criterion:forward(md:forward(trainX), trainY)
  performance.valid = criterion:forward(md:forward(validX), validY)
  
  -- save out peformance 
  local file_performance = assert( io.open(fn_performance, 'w'))
  file_performance:write( string.format('%16.4e%16.4e%16.4e\r\n', performance.train, performance.valid, performance.test) )
  file_performance:close()
  
  
  -- save model
  torch.save(fn_model, {md=md, criterion=criterion}, 'binary')
  
 
end


local function print_parameters(p, filename)

torch.save(filename, p, '')
--local file = assert( io.open(filename, 'w'))
--file:write(string.format('%16s:%d' , 'res', p.res))
--file:write(string.format('%16s:%d', 'nTrain', p.nTrain))
--file:write(string.format('%16s:%d', 'nTest', p.nTest))
--file:write(string.format('%16s:%d', 'nValid', p.nValid))
--file:write(string.format('%16s:%d', 'batchsz', p.batchsz))
--file:write(string.format('%16s:%e', 'learningRate', p.learningRate))
--file:write(string.format('%16s:%s', 'fn_evals_txt', p.fn_evals_txt))
--file:write(string.format('%16s:%s', 'fn_evals_svg', p.fn_evals_svg))
--file:write(string.format('%16s:%s', 'fn_performance', p.fn_performance))
--file:write(string.format('%16s:%s', 'fn_model', p.fn_model))
--file:write(string.format('%16s:%s', 'fn_params', p.fn_params))
--file:close()
end


return naive
