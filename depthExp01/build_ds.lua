require 'torch'
require 'OLSDataSource'

local function getPreproc(key)
  local keys= {'mean', 'middle'}

  local procs = {
    mean = function(X) 
      local N = X:size(1)
      local mu = X:mean(X:dim() - 3) 
      return mu:view(N, -1)
    end,

    middle= function(X)
      local N = X:size(1)
      local stride = X:size(2)
      local i = math.ceil(stride/2.0)
      local x = X:narrow(2, i, 1)
      return x:contiguous():view(N, -1)
    end,

    vectorize = function(X)
      local N = X:size(1)
      return X:view(N, -1)
    end
  }
  
  local mapszs = {
    mean = function(sz) 
      local n, stride, c, h, w = sz[1], sz[2], sz[3], sz[4], sz[5]
      return torch.LongStorage({n, c*h*w})
    end, 

    middle = function(sz)
      local n, stride, c, h, w = sz[1], sz[2], sz[3], sz[4], sz[5]
      return torch.LongStorage({n, c*h*w})
    end,

    vectorize = function(sz)
      local n, stride, c, h, w = sz[1], sz[2], sz[3], sz[4], sz[5]
      return torch.LongStorage({n, stride*c*h*w})
    end
  }


  return {proc=procs[key], mapsz=mapszs[key]}

end





function build_ds()
   -- data configuration
  local input  = {dtype='image', ctype='gray', res=32, step=1, stride=2, speed=2}
  local target = {dtype='depth', ctype='inverse', res=32, step=1, stride=3, speed=1}
  local inputProc =  getPreproc('vectorize')
  local targetProc = getPreproc('middle')

  local options = {
    input=input, target=target,
    inputProc=inputProc,
    targetProc=targetProc,

    splitDim=1,
    nTrain=395,
    nValid=20,
    nTest =20,
    bOnlyValid=true
  }

  return OLSDataSource(options)
end




