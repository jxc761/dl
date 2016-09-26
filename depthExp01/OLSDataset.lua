require 'torch'
local utils = require 'utils' 


local OLSDataset = torch.class('OLSDataset')

function OLSDataset:__init(data, options)

  -- unpack options 
  local defaults = {stride=1, step=1, speed=1, bidirect=false}
  local opts = utils.extend({}, defaults, options)


  local dsz = data:size()
  self.dsz = dsz
  self.S, self.T, self.F, self.C, self.W, self.H = dsz[1], dsz[2], dsz[3], dsz[4], dsz[5], dsz[6]

  local n = self.S * self.T * self.F 
  local d = self.C * self.W * self.H 
  self.data = data:view(n, d)
 


  self.stride   = opts.stride
  self.step     = opts.step
  self.speed    = opts.speed 
  self.bidirect = opts.bidirect
  self.dim =  self.C * self.W * self.H * self.stride 

  -- #scenes
  self.K = self.S
  -- #trace per scene 
  self.L = self.T * ( self.bidirect and 2 or 1)
   -- #samples per trace
  self.M = math.floor( (self.F - 1 - (self.stride-1)*self.speed + self.step ) / self.step )

  -- public
  self.nScene = self.K 
  self.nTrace = self.K * self.L
  self.nSample = self.K * self.L * self.M

  self.smp  = {self.stride, self.}

end

-- return index
--
-- idx: table/one-dimension tensor
-- 
function OLSDataset:index(idx, result)
  
  local N = utils.nElement(idx) 
  local D = self.dim
 
  result = result and result:resize(N, D) or torch.FloatTensor(N, D)
 
  for i = 1, N do
    local ii = self:mapping(idx[i])
    local x = result[i]:view(self.stride, -1) 
     x:index(self.data, 1, ii)
  end
  
  return result  
  
end

function OLSDataset:indexTraces(idx, result)
  local N = utils.nElement(idx) -- #idx -- :nElement()
  local m = self.M 
  local d = self.dim
 
  result = result and result:resize(N, m, d) or torch.FloatTensor(N, m, d)
  
  for n = 1, N do
    local x = result[n]
    local i = idx[n]
    local ii = torch.linspace((i-1)*m+1, i*m, m):long()
    self:index(ii, x)
  end
  
  return result
end


function OLSDataset:traces(idx, result)

  local N = utils.nElement(idx)
  local sz = torch.LongStorage({N, self.F, self.C, self.W})
  result = result and result:resize(sz) or torch.FloatTensor(sz)


  local org_data =  self.data:view(self.dsz)
  for n=1, N do
    local s, t = utils.ind2sub(idx[i], {self.K, self.L}) 
    

    if t <= self.T then
      result[i]:copy(org_data[s][t])
    else
      t = t - self.T 
      local trace = org_data[s][t]
      local selected = torch.range(self.F, 1, -1):long()
      result[i]:index(org_data, selected)
    end
  end


end


function OLSDataset:mapping(i)
  local sub = utils.ind2sub(i, {self.K, self.L, self.M} )
  local k, l, m = sub[1], sub[2], sub[3]

  local first, last = nil, nil
  if l <= self.T then 
    local f = 1 + (m-1) * self.step
    first = utils.sub2ind({k, l, f}, {self.S, self.T, self.F})
    last = first + (self.stride-1) * self.speed 
  else -- reverse 
    l = l - self.T 
    local f = self.F - (m-1) * self.step 
    first = utils.sub2ind({k, l, f}, {self.S, self.T, self.F})
    last = first - (self.stride-1) * self.speed
  end

  return torch.linspace(first, last, self.stride):long()
end


function OLSDataset:__tostring() 
  local t = {}
  t[1] = string.format('step=%d', self.step)
  t[#t+1] = string.format('stride=%d', self.stride)
  t[#t+1] = string.format('speed=%d', self.speed)
  t[#t+1] = string.format('bidirect=%s', self.bidirect)
  t[#t+1] = string.format('dim=%d', self.dim)
    
  t[#t+1] = string.format('nScene=%d', self.nScene) 
  t[#t+1] = string.format('nTrace=%d', self.nTrace)
  t[#t+1] = string.format('nSample=%d', self.nSample)



  t[#t+1] = string.format('#scenes=%d', self.K)
  t[#t+1] = string.format('#trace per scene=%d', self.L)
  t[#t+1] = string.format('#sample per trace=%d', self.M)
  t[#t+1] = string.format('(S, T, F, C, W, H) = (%d, %d, %d, %d, %d, %d)', 
  self.S, self.T, self.F, self.C, self.W, self.H)

  return table.concat(t, '\r\n')

end

