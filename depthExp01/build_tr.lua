require 'torch'
require 'math'

local utils = require 'utils'

local function getLR(opt)

  local min, max, method, n = utils.unpack(opt, {'min', 'max', 'method', 'n'})
  local learningrates = nil
  if method == 'log' then
    learningrates = torch.logspace(math.log10(min), math.log10(max), n)
  else
    learningrates = torch.linespace(min, max, n)
  end
  return learningrates
end


function getTrainOpts(optLR, nTrain)

  local lrs = getLR(optLR)

 -- local period  = 100
  local batchsz = 512
  local nIter   = math.floor(nTrain / batchsz)
  local sampler = torch.randperm(nIter*batchsz) % nTrain + 1
  local opts = {}

  for i=1, utils.nElement(lrs) do 

    opts[i]={
      batchsz=batchsz, 
     -- period = period, 
      nIter=nIter, 
      sgd={learningRate=lrs[i]}, 
      sampler = sampler, 
      key=string.format('%e', lrs[i])
    }

  end
  return opts
end




