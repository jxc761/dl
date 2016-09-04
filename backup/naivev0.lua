require 'torch'
require 'nn'
require 'ols'
require 'math'
require 'optim'
require 'gnuplot'
require 'image'



  

torch.setdefaulttensortype('torch.FloatTensor')

-- build model
local md = nn.Sequential()
local d = p.res * p.res
local sz = {d, math.ceil(d*1.2), math.ceil(d*1.44), math.ceil(d*1.2), d }
for i = 1, #sz-1 do
  md:add(nn.Linear(sz[i], sz[i+1]))
  md:add(nn.ReLU())
end

local criterion = nn.MSECriterion()

-- load data 
local res    = p.res
local nTrain = p.nTrain
local nTest  = p.nTest
local datadir= p.datadir
local images = ols.ReadOlsData(datadir, 'image',    'gray', res)
local depths = ols.ReadOlsData(datadir, 'depth', 'inverse', res)

local X = images:view(-1, res*res)
local Y = depths:view(-1, res*res)

local trainX = X:narrow(1, 1, nTrain)
local trainY = Y:narrow(1, 1, nTrain)
local testX  = X:narrow(1, nTrain+1, nTest)
local testY  = Y:narrow(1, nTrain+1, nTest)  


-- for debug
-- print(md)
-- print(criterion)
-- print(trainX:size())
-- print(trainY:size())
-- print(testX:size())
-- print(testY:size())

-- train model
local params, gradParams = md:getParameters()  -- flatten model parameters
local evals = {}
local nBatchs = math.floor(trainX:size(1) / p.batchsz)
local config = {learningRate = p.learningRate}
print(string.format('%16s%16s\r\n', '#epoch', 'fval') )
for epoch = 1, p.nIter do
  -- external variables: gradParams, md, criterion, trainX, trainY, p
  -- input : b:batch_index
  -- output: loss(batch), dl/dparams 
  local function feval(arg)
      
      gradParams:zero()
      local b = epoch % nBatchs - 1
      local index = b * p.batchsz + 1
      local size = p.batchsz
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
  local foutput = string.format('%s/naive_err_vs_epoch.eps', p.output)
  if epoch % p.evalPeriod == 0 then
  
    -- evaluate on the testing dataset
    local o = md:forward(testX)
    local l = criterion:forward(o, testY)
    evals[#evals+1] = l
    print( string.format('%16d\t%16.2e\r\n', epoch, l) )
    
    -- plot the evaluate result
    gnuplot.epsfigure(foutput)
    gnuplot.plot(torch.Tensor(evals))
    gnuplot.plotflush()
      
    -- evaluate on examples
    local errs = EvalModelOnExamples(md, criterion, expX, expY, prefix, h, w)
    f:write(string.)
    
  end
  

  
  
end

function EvalModelOnExamples(md, criterion, expX, expY, prefix, h, w)

c
  
  SaveImages(filenames1, estY, h, w)
  SaveImages(filenames2, diff, h, w)
  
  return errs
end

function SaveImages(filenames, X, h, w)
  local n = X:size(1)
  
  local imgs = X:view(n, 1, w, h):permute(1, 2, 4, 3)
  for i = 1, n do
    local filename = filenamse[i]
    local img = image.minmax(imgs[{i}])
    image.save(filename, img)
  end
end





