require 'optim'
function train(model, data, monitor, opt)

  local batchsz = opt.batchsz
  local nIter   = opt.nIter
  local sgdOpt  = opt.sgd
  local sampler = opt.sampler -- torch.randperm(1, nTrain)

  -- local monitor = Monitor(model, data, opt.monitor)

  monitor:start()

  local params, gradParams = model:getParameters()
  for epoch = 1, nIter do

    local function feval(s)
      local idx = sampler:narrow(1, (epoch-1)*batchsz+1, batchsz)
      local x = data:TrainX(idx)
      local y = data:TrainY(idx)
    
      return model:updateGrad(x, y)
    end
    
    
    optim.sgd(feval, params, sgdOpt)
    
    -- evaluate on the validation dataset
    monitor:monitor()

    
  end

  monitor:stop()

end




