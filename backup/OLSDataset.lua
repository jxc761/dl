require 'torch'


local OLSDataset = torch.class('OLSDataset')
---
-- OLSDataset
-- Attributes:
--   dtype,ctype
--   n
--   d
--   nbatchs
--   batchsz
--   smpsz
--  
-- Methods:
--   batch(b)
--   index(idx)
--   smps2imgs(X)
--   smp2img(x)
--   diffs2imgs(X)
--   diff2img(x)
--   
-- Private attributes:
-- org_data
-- data
---
function OLSDataset:__init(config)
  --ssi: start scene index in [1, 500], ns: number of scenes
 
  local args, org_data, ssi, ns, batchsz, dtype, ctype= xlua.unpack(
                                               {config}, 'OLSDataset', '{org_data, start_scene_idx, nscenes, batchsz, dtype, ctype}',
                                         {})
                                         
  local sz = org_data:size()
  local s, t, f, c, w, h = sz[1], sz[2], sz[3], sz[4], sz[5], sz[6]
                             
  local X = org_data:view(s*t*f , c*w*h)
  
  local offset = (ssi-1) * ( t * f) + 1
  local size   = ns * t * f                                    
  self.data = X:narrow(1, offset, size)
  self.ctype = ctype
  self.dtype = dtype
  
  self.n = self.data.size(1) -- number of samples
  self.d = self.data.size(2) -- dimension of each sample
  
  -- sample
  self.c, self.w, self.h = c, w, h
  self.s, self.t, self.f = ns, t, f
  self.smpsz = {c, w, h}

  
  self.org_data = org_data
  self.batchsz = batchsz
  self.nbatchs = floor(self.n/self.batchsz)
  
end

function OLSDataset:batch(b)
  b = (b-1) % self.nbatchs
  return self.data:narrow(1, b*self.batchsz+1, self.batchsz)
end

function OLSDataset:index(idx)
  return self.data:index(1, idx)
end


function OLSDataset:smps2imgs(X)
  local n = X:size(1)
  
  local c, w, h = unpack(self.smpsz)
  local imgs = torch.Tensor(n, c, h, w)
  for i=1, n do
    imgs[{i}]:copy( self:smp2img(X[{i}]) )
  end
  
  return imgs
  
end


function OLSDataset:smp2img(smp)
  if self.dtype == 'image' then
    return self:image2img(smp)
  elseif self.dtype == 'depth' then
    return self:depth2img(smp)
  end
end


function OLSDataset:depth2img(x)
  local img = x:view(self.smpsz):permute(1, 3, 2):clone()
  local min = img:min()
  local max = img:max() 
  img:add(-min):div(max-min+1e-5)
  return img
end

function OLSDataset:image2img(x)
  local img = x:view(self.smpsz):permute(1, 3, 2):clone()
  return img
end

function OLSDataset:diffs2imgs(X)

  local n = X:size(1)
  
  local c, w, h = unpack(self.smpsz)
  local imgs = torch.Tensor(n, c, h, w)
  for i=1, n do
    imgs[{i}]:copy( self:diff2img(X[{i}]) )
  end 
  return imgs
end

function OLSDataset:diff2img(x)
  local img = x:view(self.smpsz):permute(1, 3, 2):clone()
  local min = img:min()
  local max = img:max() 
  img:add(-min):div(max-min+1e-5)
  return img
end


function OLSDataset:tostring()
  local template = [[
    dtype      : %s
    ctype      : %s
    size       : %dx%dx%dx%dx%dx%d(#scenes x #traces x #frames x #channel x width x height) 
    #samles    : %d
    dimension  : %d
  ]]
  
  return string.format(template, self.dtype, self.ctype, self.s, self.t, self.f, self.c, self.w, self.h, self.n, self.d)
end



