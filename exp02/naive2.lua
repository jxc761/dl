
require 'torch'
require 'optim'
require 'cunn'
require 'cutorch'

require 'Monitor'
local ols = require 'ols'

local naive2={}


local function save_params(filename, p)

  local file = assert( io.open(filename, 'w'))
  
  file:write(string.format('%16s=%s\r\n', 'input.dtype', p.input.dtype) )
  file:write(string.format('%16s=%s\r\n', 'input.ctype', p.input.ctype) )
  file:write(string.format('%16s=%d\r\n', 'input.res', p.input.res))

  file:write(string.format('%16s=%s\r\n', 'target.dtype', p.target.dtype) )
  file:write(string.format('%16s=%s\r\n', 'target.ctype', p.target.ctype) )
  file:write(string.format('%16s=%d\r\n', 'target.res', p.target.res))

  
  file:write(string.format('%16s=%d\r\n', 'nTrain', p.nTrain))
  file:write(string.format('%16s=%d\r\n', 'nTest', p.nTest))
  file:write(string.format('%16s=%d\r\n', 'nValid', p.nValid))
  
  file:write(string.format('%16s=%s\r\n', 'struct', p.struct)
  file:write(string.format('%16s=%d\r\n', 'nhidden', p.nhidden)
  file:write(string.format('%16s=%f\r\n',  'alpha', p.alpah)
  
  
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


local function Dataset(config)
	
	local input = config.input
	local target = config.target
	local nTrain = config.nTrain
	local nTest = config.nTest 
	local nValid = config.nValid

	local X = ols.LoadDataset(input.dtype, input.ctype, input.res)
	local Y = ols.LoadDataset(target.dtype, target.ctype, target.res)
  X = X:view(-1, input.res*input.res)
	Y = Y:view(-1, target.res*target.res)
	
	X = X:cuda()
	Y = Y:cuda()
	
	local trainX = X:narrow(1, 1, nTrain)
  local trainY = Y:narrow(1, 1, nTrain)
  local validX = X:narrow(1, nTrain+1, nValid)
  local validY = Y:narrow(1, nTrain+1, nValid)  
  local testX  = X:narrow(1, nTrain+nValid+1, nTest)
  local testY  = Y:narrow(1, nTrain+nValid+1, nTest)  

  return  train ={X=trainX, Y= trainY}, valid={X=validX, Y=validY}, test={testX, testY}
  

end


local function Model(config)
	local d = config.dim
	local a = config.alpha
	local struct = config.struct 
	local nhidden = config.nhidden

	local sztable1 = {
		['3'] = {d, d*a, d*a*a, d*a, d}, 
		['4'] = {d, d*a, d*a*a, d*a*a, d*a, d},
		['5'] = {d, d*a, d*a*a, d*a*a*a, d*a*a, d*a, d}
	}

	local sztable2 = {
		['3'] = {d, d*a, d*a*a, d*a*a*a, d},
		['4'] = {d, d*a, d*a*a, d*a*a*a, d},
		['5'] = {d, d*a, d*a*a, d*a*a*a, d*a*a*a*a, d}
	}

	local sztable = { sym=sztable1, inc= sztable2}
	local sz = sztable[struct][nhidden]
  
  local md = nn.Sequential()
  for i = 1, nhidden+1 do
    md:add(nn.Linear( math.ceil(sz[i]), math.ceil(sz[i+1]) ))
    md:add(nn.ReLU())
  end

  local criterion = nn.MSECriterion()
  
  md:cuda()
  criterion:cuda()
  
  return {md=md, criterion=criterion}

end



local function forward(model, X, Y)
	local step=1000
  local sum = 0

  local n = X:size(1)
  for index = 1, n, step do
  	local size = step < (n-index+1) and step or (n-index+1) 
    local x = X:narrow(1, index, size)
    local y = Y:narrow(1, index, size)
    local l = model.criterion:forward(model.md:forward(x), y)
    sum = sum + l * size
  end
  result = sum / n;
  return result
end



function naive2.run(p)
	torch.manualSeed(0)
  torch.setdefaulttensortype('torch.FloatTensor')
  
  -- save parameters
  if p.fn_parameters then
    save_params(p.fn_parameters, p)
  end
  
	local train, valid, test = Datasets(p)
	local model = Model(p)
	local monitor = Monitor(p)

  local nIter = p.nIter
	local batchsz = p.batchsz
	local learningRate = p.learningRate



  local nTrain = train.X:size(1)
  local shuffle = torch.randperm(nTrain).

  sys.tic() -- start timer
  monitor.start()
  
	local params, gradParams = model.md:getParameters() 
  for epoch = 1, nIter do
    -- external variables: gradParams, md, criterion, trainX, trainY, p
    -- input : b:batch_index
    -- output: loss(batch), dl/dparams 
    local function feval(s)
      gradParams:zero()
      local b = (epoch-1) % nBatchs 
      local offset = b * batchsz + 1
      local size = batchsz
      
      local idx = shuffle:narrow(1, offset, size)
      local x = train.X:index(idx) 
      local y = train.Y:index(idx)
    

      local o = model.md:forward(x)               -- output of the network   
      local l = model.criterion:forward(o, y)     -- loss of the model
      local dl  = model.criterion:backward(o, y) -- d_loss/d_output
      model.md:backward(x, dl)                -- d_loss/d_parameters and d_loss / d_x
      return l, gradParams
    end
    
    
    optim.sgd(feval, params, {learningRate = learningRate})
    
    -- evaluate on the validation dataset
    if epoch % p.evalPeriod == 0 then
      local duration = sys.toc()      
      local validl = forward(model, valid) 
      monitor:monitor(epoch, duration, validl)
    end
  end
  
  monitor:stop()
  
  -- evaluate peformance
  local perform = {}
  perform.duration = sys.toc() 
  perform.valid = forward(model, valid) 
  perform.test  = forward(model, test) 
  perform.train = forward(model, train)
  
  -- save out peformance 
  if  fn_performance then
    local file_performance = assert( io.open(fn_performance, 'w'))
    file_performance:write('learning_rate\tduration\ttraining\tvalidation\ttesting\r\n') 
    file_performance:write( string.format('%e\t%e\t%e\t%e\t%e\r\n', learningRate, perform.duration, perform.train, perform.valid, perform.test) )
    file_performance:close()
  end
  
  -- save model
  if fn_model ~= nil then
    torch.save(fn_model, model, 'binary')
  end
  
  return {model=model, perform=perform, process=monitor.record()}
  
end

return naive2