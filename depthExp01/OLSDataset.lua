require 'torch'
local utils = require 'utils' 


local OLSDataset = torch.class('OLSDataset')

function OLSDataset:__init(data, options)

  -- unpack options 
  local defaults = {stride=1, step=1, speed=1, bidirect=false}
  local opts = utils.extend({}, defaults, options)


  local dsz = data:size()
  self.dsz = dsz
  self.S, self.T, self.F, self.C, self.H, self.W = dsz[1], dsz[2], dsz[3], dsz[4], dsz[5], dsz[6]

  local n = self.S * self.T * self.F 
  local d = self.C * self.W * self.H 
  self.data = data:view(n, self.C, self.H, self.W)
 


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
  self.nScene  = self.K 
  self.nTrace  = self.K * self.L
  self.nSample = self.K * self.L * self.M

  self.smpsz = torch.LongStorage({self.stride, self.C, self.H, self.W})
  self.frmsz = torch.LongStorage({self.C, self.H, self.W})
end

-- return 
-- N x stride x c x h x w
-- idx: table/one-dimension tensor
-- 
function OLSDataset:index(idx, result)
  
  local N = utils.nElement(idx) 
  local sz = torch.LongStorage{N, self.stride, self.C, self.H, self.W}
  result = result and result:resize(sz) or torch.FloatTensor(sz)--:cuda()
  -- result:cuda()
 
  for i = 1, N do
    local ii = self:mapping(idx[i])
    result[i]:index(self.data, 1, ii)
  end
  
  return result  
  
end

-- return 
-- N x #samples per trace x stride x c x h x w
-- 
function OLSDataset:indexTraces(idx, result)
  local N = utils.nElement(idx) -- #idx -- :nElement()
  local sz = torch.LongStorage{N, self.M, self.stride, self.C, self.H, self.W }
  local m = self.M
  result = result and result:resize(sz) or torch.FloatTensor(sz)--:cuda()
  
  for n = 1, N do
    local x = result[n]
    local i = idx[n]
    local ii = torch.linspace((i-1)*m+1, i*m, m):long()
    self:index(ii, x)
  end
  
  return result
end

function OLSDataset:indexTrace(i, result)
  local sz = torch.LongStorage{self.M, self.stride, self.C, self.H, self.W }
  result = result and result:resize(sz) or torch.FloatTensor(sz)--:cuda()
  local ii = torch.linspace((i-1)*m+1, i*m, m):long()
  self:index(ii, result)
  return result
end

function OLSDataset:traces(idx, result)
  
  local N  = utils.nElement(idx)
  local sz = torch.LongStorage{N, self.F, self.C, self.H, self.W}
  result = result and result:resize(sz) or torch.FloatTensor(sz)
  
  local org_data =  self.data:view(self.dsz)
  for i=1, N do
    local s, t = utils.ind2sub(idx[i], {self.K, self.L})  
    if t <= self.T then
      result[i]:copy(org_data[s][t])
    else
      t = t - self.T 
      local trace = org_data[s][t]
      local selected = torch.range(self.F, 1, -1):long()
      result[i]:index(trace, 1, selected)
    end
  end

  return result

end

function OLSDataset:trace(i, result)
  local sz = torch.LongStorage{self.F, self.C, self.H, self.W}
  result = result and result:resize(sz) or torch.FloatTensor(sz)--:cuda()

  local org_data =  self.data:view(self.dsz)
  local s, t = utils.ind2sub(i, {self.K, self.L})  
  if t <= self.T then
    result:copy(org_data[s][t])
  else
    t = t - self.T 
    local trace = org_data[s][t]
    local selected = torch.range(self.F, 1, -1):long()
    result:index(trace, 1, selected)
  end

  return result

end



function OLSDataset:mapping(i)

  local k, l, m = utils.ind2sub(i, {self.K, self.L, self.M} )

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



  t[#t+1] = string.format('(K, L, M) = (%d, %d, %d)', self.K, self.L, self.M)
  t[#t+1] = string.format('(S, T, F) = (%d, %d, %d)', self.S, self.T, self.F)
  t[#t+1] = string.format('(C, H, W) = (%d, %d, %d)', self.C, self.H, self.W)

  return table.concat(t, '\r\n')

end

