--- the performance of using mean
require 'nn'
local ols=require 'ols'


torch.setdefaulttensortype('torch.FloatTensor')


local p = {      
  input  = {dtype = 'image',  ctype='gray', res=16}, 
  target = {dtype= 'depth', ctype='inverse', res=16},
  nTrain = 400 * 40 * 30,
  nValid = 50 * 40 * 30,
  nTest  = 50 * 40 * 30,
  batchsz = 128,
  learningRate = 0.07,
  evalPeriod = 100,
  nIter = 10000,
  usegpu=false,
  fn_evals_txt   = string.format('%s/evals.txt', output),
  fn_evals_svg   = string.format('%s/errs_vs_epoch.svg', output),
  fn_performance = string.format('%s/performance.txt', output),
  fn_model       = nil, --string.format('%s/model.dat', output),
  fn_parameters  = string.format('%s/parameters.txt', output)
  } 
  
 --run(param) 
    
    
  local input = p.input
  local target = p.target
  
  local nTrain, nValid, nTest = p.nTrain, p.nValid, p.nTest
  
  local usegpu = p.usegpu
  
  
  --- load data in
  local X = ols.LoadDataset(input.dtype, input.ctype, input.res)
  local Y = ols.LoadDataset(target.dtype, target.ctype, target.res)
  X = X:view(-1, input.res*input.res)
  Y = Y:view(-1, target.res*target.res)
 
  if usegpu then
    X = X:cuda()
    Y = Y:cuda()
  end

  local trainX = X:narrow(1, 1, nTrain)
  local trainY = Y:narrow(1, 1, nTrain)
  local validX = X:narrow(1, nTrain+1, nValid)
  local validY = Y:narrow(1, nTrain+1, nValid)  
  local testX  = X:narrow(1, nTrain+nValid+1, nTest)
  local testY  = Y:narrow(1, nTrain+nValid+1, nTest)  
  
  
  local mu = torch.mean(trainY, 1)
 
  local criterion = nn.MSECriterion()
  local ones
  local perform = {} 
  
 
  ones = torch.ones(nTest, 1) 
  perform.test = criterion:forward( ones * mu, testY)
  
  ones = torch.ones(nValid, 1) 
  perform.valid = criterion:forward(ones * mu, validY)
  
  
  ones = torch.ones(nTrain, 1) 
  perform.train = criterion:forward(ones * mu, trainY)
  
  
  print(string.format('Train: %f, Valid: %f, Test: %f\r\n', perform.train, perform.valid, perform.test))
  -- print(string.format('Train: %e, Valid: %e, Test: %e\r\n', perform.train, perform.valid, perform.test))