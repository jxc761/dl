local function train(model, data, monitor, opt)

  local batchsz = opt.batchsz
  local nIter   = opt.nIter
  local sgdOpt  = opt.sgd
  local sampler = opt.sampler -- torch.randperm(1, nTrain)
  
  monitor:start()

  local params, gradParams = model:getParameters()
  for epoch = 1, nIter do

    local function feval(s)
      local idx = sampler:narrow(1, (epoch-1) * batchsz, batchsz)
      local x = data.trainX(idx) 
      local y = data.trainY(idx)      
      return model:updateGrad(x, y)
    end
    
    
    optim.sgd(feval, params, {learningRate = lr})
    
    -- evaluate on the validation dataset
    monitor:monitor()

  end

  monitor:stop()


end

