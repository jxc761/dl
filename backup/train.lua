require 'optim'

local function train(md, criterion, trainX, trainY, config)

  local nIter        = config.nIter
  local learningRate = config.learningRate
  local monitor      = config.monitor
  

  monitor.start()
  local params, gradParams = md:getParameters()  -- flatten model parameters
  for epoch = 1, nIter do
  
    -- external variables: gradParams, md, criterion, trainX, trainY, epoch
    -- input : not important
    -- output: loss(batch), dl/dparams 
    local function feval(arg)
        
      gradParams:zero()
      
      local x = trainX:batch(epoch)
      local y = trainY:batch(epoch)
      
      local o = md:forward(x)               -- output of the network   
      local l = criterion:forward(o, y)     -- loss of the model
      local dl  = criterion:backward(o, y)  -- d_loss/d_output
      md:backward(x, dl)                    -- d_loss/d_parameters and d_loss / d_x
      
      return l, gradParams
         
    end
  
    
    optim.sgd(feval, params, {learningRate = learningRate})
    
    -- evaluate
    monitor.monitor(epoch, md, criterion)
    
  end -- end for
  
  monitor.stop()
  
end

return train