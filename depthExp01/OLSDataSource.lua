require 'torch'
require 'OLSDataset'

local ols = require 'ols'

local OLSDataSource=torch.class('OLSDataSource')



local function ValidSceneIdx(N)
  
  local scene_with_undefined_depth = torch.LongTensor{
    2, 3, 4, 18, 25, 28, 32, 58, 70, 72, 75, 
    79, 96, 103, 104, 126, 128, 137, 143, 155, 
    156, 172, 173, 176, 194, 234, 243, 247, 250, 
    275, 289, 303, 315, 334, 342, 352, 353, 361, 
    362, 369, 373, 375, 389, 394, 404, 406, 408, 
    415, 419, 423, 438, 439, 443, 451, 452, 454, 
    455, 456, 461, 462, 473, 487, 492, 493, 498
  }
  local undefined = scene_with_undefined_depth[scene_with_undefined_depth:le(N)]
  local mask = torch.ByteTensor(N, 1):fill(1)
  mask:indexFill(1, undefined, 0)
  
  local idx = torch.LongTensor(N, 1)
  local i = 0
  idx:apply(function()
    i = i + 1
    return i
  end)

  return idx[mask]
end

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
  local nTest  = splitOpt.nTest and splitOpt.nTest or ( data:size(1)-nTrain-nValid )
  
  
  local train = data:narrow(splitDim, 1, nTrain)
  local valid = data:narrow(splitDim, nTrain+1, nValid)
  local test  = data:narrow(splitDim, nTrain+nValid+1, nTest)
  
  local trainset = OLSDataset(train,  dataOpt) 
  local validset = OLSDataset(valid,  dataOpt ) 
  local testset  = OLSDataset(test,   dataOpt ) 
  
  return {trainset, validset, testset}
end




--- use to build the data source
-- Usage:
--  
-- input: 
-- options = {
--   input =  {dtype, ctype, res, step, stride, speed, bidirect, preprocess={func, dimOut}}
--   target = {dtype, ctype, res, step, stride, speed, bidirect, preprocess={func, dimOut}}
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
  
  local splitOpt = utils.getfields(opts, {splitDim, nTrain, nValid, nTest}) 
  
  local X = ols.LoadDataset(input.dtype, input.ctype, input.res)
  local Y = ols.LoadDataset(target.dtype, target.ctype, target.res)
  
  -- rule out the scenes with undefined depth
  if opts.bExcludeUndefined then
    local N = X:size(1)
    local idx = ValidSceneIdx(N)
    X = X[idx]
    Y = Y[idx]
  end


  self.trainX, self.validX, self.testX = splitDataset(X, splitOpt, dsXOpt)
  self.trainY, self.validY, self.testY = splitDataset(Y, splitOpt, dsYOpt)
  
  self.nTrain = self.trainX.nSample
  self.nValid = self.validX.nSample
  self.nTest = self.testX.nSample
  self.nTotal = self.nTrain+self.nValid + self.nTest 

  -- self.szx = self.trainX.stride 

  self.yproc = function(data) {
    data:view(data:size(1). self.yopt.stride, )
  }

  
end


function OLSDatasSource:TrainX(idx)
  idx = idx and idx or torch.range(1, self.nTrain):long()
  return self.trainX:index(idx)
end

function OLSDataSource:TrainY(idx)
  idx = idx and idx or torch.range(1, self.nTrain):long()
  return self.trainY:index(idx)
end

function OLSDataSource:ValidX(idx)
  idx = idx and idx or torch.range(1, self.nValid):long()
  return self.validX:index(idx)
end


function OLSDataSource:ValidY(idx)
  idx = idx and idx or torch.range(1, self.nValid):long()
  return self.validY:index(idx)
end

function OLSDataSource:TestX(idx)
  idx = idx and idx or torch.range(1, self.nTest):long()
  return self.testX:index(idx)
end

function OLSDataSource:TestY(idx)
  idx = idx and idx or torch.range(1, self.nTest):long()
  return self.testY:index(idx)
end

function OLSDataSouce:__ind(idx, sz)

  local N = utils.nElement(idx)
  local s1, s2, s3=sz[1], sz[1]+sz[2], sz[1]+sz[2]+sz[3]
  local c1, c2, c3=0
  local idx1, idx2, idx3 ={}, {}, {}

  for i=1, N do
    if idx[i] <= s1 then
      c1 = c1 + 1
      idx1[c1] = idx[i]

    elseif idx[i] <= s2 then
      c2 = c2 + 1
      idx2[c2] = idx[i] - s1
    elseif idx[i] <= s3 then 
      c3 = c3+1
      idx3[c3] = idx[i] - s2
    else
      error('the index is out of boundary')
    end
  end
  return idx1, idx2, idx3  
end


function OLSDataSource:X(idx)
  local idx1, idx2, idx3 = self.__ind(idx, {self.nTrain, self.nValid, self.nTest})
  local X1 = self:TrainX(idx1)
  local X2 = self:ValidX(idx2)
  local X3 = self:TestX(idx3)
  return torch.concat(X1, X2, X3, 1)

end

function OLSDataSource:Y(idx)
  local idx1, idx2, idx3 = self.__ind(idx, {self.nTrain, self.nValid, self.nTest})
  local X1 = self:TrainY(idx1)
  local X2 = self:ValidY(idx2)
  local X3 = self:TestY(idx3)
  torch.concat(X1, X2, X3, 1)

end

function OLSDataSource:TraceX(idx)


end


function OLSDatasource:TraceY(idx)

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
