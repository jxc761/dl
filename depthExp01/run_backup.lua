
require'build_ds'
require'build_md'
require'Examples'
require'Monitor'

require'paths'
local utils=require'utils'

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


local function getTrainOpts(nTrain, root_output)
  local lrs = getLR({min=1e-3, max=1, n=4, method='log'})

  local period  = 100
  local batchsz = 128
  local nIter   = math.floor(nTrain / batchsz)
  local sampler = torch.randperm(nIter*batchsz) % nTrain + 1
  local opts = {}

  for i=1, utils.nElement(lrs) do 
    -- local fn_proc     = string.format('%s/proc_%d.txt', root_output, i)
    -- local fn_proc_exp = string.format('%s/proc_%d.dat', root_output, i) 
    -- local fn_result   = string.format('%s/result_%d.dat', root_output, i)
    -- local monitor = {fn_proc=fn_proc, fn_proc_exp=fn_proc_exp}

    opts[i]={batchsz=batachsz, period = period, nIter=nIter, 
    sgdOpt={learningRate=lrs[i]}, 
    sampler = sampler, 
    -- monitor = monitor,
    key=string.format('%e', lrs[i])}

  end

    -- local fn_proc_exp = string.format('%s/proc_%d.dat', root_output, i) 
    -- local fn_result   = string.format('%s/result_%d.dat', root_output, i)
    -- local monitor = Monitor(model, data, examples, {fn_proc=fn_proc, fn_proc_exp=fn_proc_exp})
  return opts
end


local function batch_mse_ds(model, data, dataset)
  local N = data:numb(dataset)
  local batchsz=1200
  local B = math.ceil(N/batchsz)

  local n=0
  local e=0

  for b = 1, B do
    print('processing ' .. b)
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

local function evaluate_on_exps(model, examples, fn_output)

  local predSmpY = model:predict(examples:smpX())

  local X = examples:trcX()
  local Y = examples:trcY()
  local predY = {}
  for i=1, examples.nTrace do 
    predY[i]= model:predict(X[i])
  end

  local predTrcY = utils.concat(1, predY)

  if fn_output then
    local f = torch.DiskFile(fn_output, 'w'):binary()
    f:writeObject(predSmpY)
    f:writeObject(predTrcY)
    f:close()
  end

  return predSmpY, predTrcY

end

local function evaluate(model, data, examples, fmse, fn_result)
  local mse = evaluate_on_ds(model, data)
  local predSmpY, predTrcY=evaluate(model, data, examples)
  return mse, predSmpY, predTrcY
end



local function test_model(model, data, examples, trainOpts, root_output)

  local fpeform = assert( io.open(string.format('%s/peform.txt', root_output), 'w'))
  local fn_examples= string.format('%s/examples.dat', root_output)
  examples:save(fn_examples)

  for i=1, #trainOpts do

    -- setup monitor
    -- local fn_proc     = string.format('%s/proc_%d.txt', root_output, i)
    -- local fn_proc_exp = string.format('%s/proc_%d.dat', root_output, i) 
    local fn_result   = string.format('%s/result_%d.dat', root_output, i)
    local monitor = Monitor(model, data, examples, trainOpts[i].monitor)


    model:reset()
    
    sys.tic()
    ---train(model, data, trainOpts[i])
    duration = sys.toc()

    evaluate_on_exps(model, examples, fn_result)
    
    local mse = evaluate_on_ds(model, data)
    fpeform:write(string.format('%s\t%e\t%e\t%e\t%e\r\n', trainOpts[i].key, duration, mse.train, mse.valid, mse.test))  
    fpeform:flush()

  end

  fpeform:close()
end

torch.manualSeed(0)
torch.setdefaulttensortype('torch.FloatTensor')

local data     = build_ds()
local model    = build_md()
local examples = Examples(data)

print(data)
print(model)

local root_output = string.format('%s/depthExp01/model1_step01', utils.buffer_dir())
paths.mkdir(root_output)
local trainOpts = getTrainOpts(data.nTrain, root_output)

test_model(model, data, examples, trainOpts, root_output)
