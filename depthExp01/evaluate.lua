require 'torch'

local utils=require'utils'

local function batch_mse_ds(model, data, dataset)
  local N = data:numb(dataset)
  local batchsz=1200
  local B = math.ceil(N/batchsz)

  local n=0
  local e=0

  for b = 1, B do
    --print('processing ' .. b)
    local first =  (b-1)*batchsz+1
    local last  =  b*batchsz > N and N or  b*batchsz 
    local idx = torch.range(first, last):long()
    local X = data:indexX(idx, dataset)
    local Y = data:indexY(idx, dataset)
    local YY = model:predict(X)
    local d = Y - YY
    n = n + d:nElement()
    e = e + d:pow(2):sum()
  end 

  return math.sqrt(e/n)
end



local function evaluate_on_ds(model, data)

  local batchsz=1200
  local mse = {}
  mse.train =  batch_mse_ds(model, data, 'train')
  mse.test  =  batch_mse_ds(model, data, 'test')
  mse.valid =  batch_mse_ds(model, data, 'valid') 
  return mse 
end

local function evaluate_on_exps(model, examples)

  local predSmpY = model:predict(examples:smpX())

  local X = examples:trcX()
  local Y = examples:trcY()
  local predY = {}
  for i=1, examples.nTrace do 
    predY[i]= model:predict(X[i])
  end

  local predTrcY = utils.concat(1, predY)
  return predSmpY:float(), predTrcY:float()
end

function evaluate(model, data, examples)

  local predSmpY, predTrcY  = evaluate_on_exps(model, examples)
  local mse = evaluate_on_ds(model, data)

  return mse, predSmpY, predTrcY
end
