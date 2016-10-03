require 'torch'



require 'OLSDataset'
local ols = require 'ols'
local utils = require 'utils'

local OLSDataSource=torch.class('OLSDataSource')


--- 
-- construct train, valid, and test datasets
-- Usage:
-- Input:
-- opts = { dtype, ctype, res, step, speed, stride, bidirect, splitDim, nTrain, nTest, nValid}
-- Output 
-- {trainset, validset, testset}
-- 
local function splitDataset(data, splitOpt, dataOpt)

  local splitDim = splitOpt.splitDim
  local nTrain = splitOpt.nTrain
  local nValid = splitOpt.nValid
  local nTest  = splitOpt.nTest and splitOpt.nTest or ( data:size(splitDim)-nTrain-nValid )
  
  print(splitDim)
  print(nTrain)
  print(nValid)
  print(nTest)
  print(data:size())

  local train = data:narrow(splitDim, 1, nTrain)
  local valid = data:narrow(splitDim, nTrain+1, nValid)
  local test  = data:narrow(splitDim, nTrain+nValid+1, nTest)
  
  local trainset = OLSDataset(train,  dataOpt) 
  local validset = OLSDataset(valid,  dataOpt ) 
  local testset  = OLSDataset(test,   dataOpt ) 
  
  return trainset, validset, testset
end





--- use to build the data source
-- Usage:
--  
-- input: 
-- options = {
--   input =  {dtype, ctype, res, step, stride, speed, bidirect}
--   target = {dtype, ctype, res, step, stride, speed, bidirect}
--
--   inputProc= {func=, dimOut=} x = func(s), s[N, stride, c, h, w] 'noramlize|middle|mean'
--   targetProc={func=, dimOut=} y = func(s),normalize|middle|mean'
--
--   splitDim
--   nTrain
--   nValid
--   nTest = nil
--   
--   bExcludeUndefined
--
-- }
-- 
-- output
-- 
-- datasource = {
--   
--   train={X, Y}
--   valid={X, Y}
--   test= {X, Y}
-- 
-- }

function OLSDataSource:__init(opts)
  
  local input = opts.input
  local target = opts.target
  
  local X = ols.LoadDataset(input.dtype, input.ctype, input.res)
  local Y = ols.LoadDataset(target.dtype, target.ctype, target.res)
  
  -- rule out the scenes with undefined depth
  if opts.bOnlyValid then
    local idx = ols.ValidSceneIdx()
    X = X:index(1, idx)
    Y = Y:index(1, idx)
  end
  

  local splitOpt = utils.getfields(opts, {'splitDim', 'nTrain', 'nValid', 'nTest'}) 
  local dsXopt   = utils.getfields(opts.input, {'step', 'stride', 'speed', 'bidirect'})
  local dsYopt   = utils.getfields(opts.target, {'step', 'stride', 'speed', 'bidirect'})

  self.trainX, self.validX, self.testX = splitDataset(X, splitOpt, dsXopt)
  self.trainY, self.validY, self.testY = splitDataset(Y, splitOpt, dsYopt)
  
  self.nTrain = self.trainX.nSample
  self.nValid = self.validX.nSample
  self.nTest  = self.testX.nSample
  self.nTotal = self.nTrain+self.nValid + self.nTest 

  self.nTrainTrace = self.trainX.nTrace
  self.nValidTrace = self.validX.nTrace
  self.nTestTrace  = self.testX.nTrace
  self.nTrace      = self.nTrainTrace + self.nValidTrace + self.nTestTrace 

  self.xproc = opts.inputProc.proc
  self.yproc = opts.targetProc.proc
  -- self.xproc, self.xszmap = inputProc and getPreproc(inputProc) 
  -- self.yproc, self.yszmap = targetProc and getPreproc(targetProc) 

  -- self.xsrcsz = self.TrianX.smpsz:clone()
  -- self.ysrcsz = self.TrainY.smpsz:clone()
  -- self.xsz = self.xszmap(self.xsrcsz)
  -- self.ysz = self.yszmap(self.ysrcsz) 

  self.dataX = {
    train = self.trainX,
    valid = self.validX,
    test = self.testX
  }

  self.dataY = {
    train = self.trainY,
    valid = self.validY,
    test = self.testY
  }
  self.numbs = {
    train = self.nTrain,
    valid = self.nValid,
    test  = self.nTest
  }

end



function OLSDataSource:TrainX(idx)
  idx = idx or torch.range(1, self.nTrain):long()
  return self:indexX(idx, 'train')
end

function OLSDataSource:TrainY(idx)
  idx = idx or torch.range(1, self.nTrain):long()
  return self:indexY(idx, 'train')
end

function OLSDataSource:ValidX(idx)
  idx = idx or torch.range(1, self.nValid):long()
  return self:indexX(idx, 'valid')
end


function OLSDataSource:ValidY(idx)
  idx = idx and idx or torch.range(1, self.nValid):long()
  return self:indexY(idx, 'valid')
end

function OLSDataSource:TestX(idx)
  idx = idx and idx or torch.range(1, self.nTest):long()
  return self:indexX(idx, 'test')
end

function OLSDataSource:TestY(idx)
  idx = idx and idx or torch.range(1, self.nTest):long()
  return self:indexY(idx, 'test')
end


function OLSDataSource:indexX(idx, set)
  return self.xproc(self.dataX[set]:index(idx))
end

function OLSDataSource:indexY(idx, set)
  return self.yproc(self.dataY[set]:index(idx))
end

function OLSDataSource:numb(set)
  return self.numbs[set]
end

function OLSDataSource:TracesX(idx, set)
  local trace = self.dataX[set]:indexTraces(idx)
  local sz =  trace:size()
  local t, m, stride, c, h, w =sz[1], sz[2], sz[3], sz[4], sz[5], sz[6]
  local X = self.xproc(trace:view(t*m, stride, c, h, w))
  return X:view(t, m , -1)
end

function OLSDataSource:TracesY(idx, set)
  local trace = self.dataY[set]:indexTraces(idx)
  local sz =  trace:size()
  local t, m, stride, c, h, w =sz[1], sz[2], sz[3], sz[4], sz[5], sz[6]
  local Y = self.yproc(trace:view(t*m, stride, c, h, w))
  return Y:view(t, m, -1)
end




function OLSDataSource:traces(idx, set)
  local X = self.dataX[set]:traces(idx)
  local Y = self.dataY[set]:traces(idx)
  return X, Y
end

function OLSDataSource:__tostring()
  local t = {}

  t[#t+1] = 'TrainX'
  t[#t+1] = self.trainX:__tostring()
  t[#t+1] = 'TrainY'
  t[#t+1] = self.trainY:__tostring()
  t[#t+1] = 'ValidX'
  t[#t+1] = self.validX:__tostring()
  t[#t+1] = 'ValidY'
  t[#t+1] = self.validX:__tostring()
  t[#t+1] = 'testX'
  t[#t+1] = self.testX:__tostring()
  t[#t+1] = 'testY'
  t[#t+1] = self.testY:__tostring()

  return table.concat(t, '\r\n')
end



--[[
--   
--   expSmpIdx | nExpSmp
--   trainExpSmpIdx | nTrainExpSmp
--   validExpSmpIdx | nValidExpSmp
--   testExpSmpIdx  | nTestExpSmp
--  
--   expTrcIdx | nExpTrc
--   trainExpTrcIdx   | nTrainExpTrc
--   validExpTrcIdx   | nValidExpTrc
--   testExpTrcIdx    | nTextExpTrc
--   
--   trainExpSmp={X, Y, idx}
--   validExpSmp={X, Y, idx}
--   testExpSmp ={X, Y, idx}
--   
--   trainExpTrc = {X, Y, idx}
--   validExpTrc = {X, Y, idx}
--   testExpTrc = {X, Y, idx}
--   
--   
--   
--
local function construct_dataset(data, opts)

  local dataset = OLSDataset( utils.extend({}, opts, {data=data}) ) 
  local nExpSmp = opts.nExpSample
  local nExpTrc = opts.nExpTrace
  
  local expSmpIdx = torch.Tensor(1, nExpSmp):random(1, dataset.nSample)
  local expTrcIdx = torch.Tensor(1, nExpTrc):random(1, dataset.nTrace)
  
  local expSmp = dataset:index(expSmpIdx)
  local expTrc = dataset:indexTrace(expTrcIdx)
  
  return {dataset, expSmpIdx, expSmp, expTrcIdx, expTrc}

end

]] 
