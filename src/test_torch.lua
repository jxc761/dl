require 'torch'
require 'nn'
require 'math'
require 'optim'
require 'gnuplot'
require 'paths'

local ols = require 'ols'
local os = paths.uname()

torch.setdefaulttensortype('torch.FloatTensor')

 p = {
    res = 16,
    nTrain = 450 * 40 * 30,
    nTest  = 50 * 40 * 30,
    
    batchsz = 128,
    learningRate = 0.01,
    evalPeriod = 10,
    nIter = 1000,
    
    output = './buffer'
}
  
if os == 'Darwin' then
  p = {
    res = 16,
    nTrain = 40 * 40 * 30,
    nTest  = 5 * 40 * 30,
    
    batchsz = 128,
    learningRate = 0.001,
    evalPeriod = 10,
    nIter = 1000,
    
    output = './buffer'
  }
end


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
local images = ols.LoadDataSet('image',    'gray', res)
local depths = ols.LoadDataSet('depth', 'inverse', res)

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
      local b = (epoch-1) % nBatchs 
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
  local foutput = string.format('%s/naive_err_vs_epoch.svg', p.output)
  if epoch % p.evalPeriod == 0 then
  
    -- evaluate on the testing dataset
    local o = md:forward(testX)
    local l = criterion:forward(o, testY)
    evals[#evals+1] = l
    print( string.format('%16d\t%16.2e\r\n', epoch, l) )
    
    -- plot the evaluate result
    if os == 'Darwin' then
      gnuplot.setgnuplotexe('/usr/local/bin/gnuplot')
    end
      
    gnuplot.setterm('svg')
    --gnuplot.epsfigure(foutput)
    gnuplot.svgfigure(foutput)
    gnuplot.plot(torch.Tensor(evals))
    gnuplot.plotflush()
      
  end
 
  
  
end





