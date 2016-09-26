require 'torch'
require 'paths'

-- local torch = require 'torch'
-- local utils = require 'utils'

local function CacheDir()
  local osname = paths.uname()
  
  local datadir = {
    Darwin = '/Users/Jing/Dropbox/dev/dl/buffer/cache',
    Linux  = '/home/jxc761/dl/buffer/cache'
  }
   
  return datadir[osname]
  
end

local function DatasetFilename(dtype, ctype, res)
  
  local key = string.format('%s_%s_%dx%d',dtype, ctype, res, res)
  
  
  local cachedir = CacheDir()
  local filename = string.format('%s/%s.cache', cachedir, key)
  
  return filename
  
end


local function DatasetSize(dtype, ctype, res)

  local key = string.format('%s_%s_%dx%d',dtype, ctype, res, res)
  
  -- size
  local sztable = { 
      image_gray_16x16    = {500, 40, 30, 1, 16, 16}, 
      image_gray_32x32    = {500, 40, 30, 1, 32, 32},
      depth_normal_16x16  = {500, 40, 30, 1, 16, 16},
      depth_normal_32x32  = {500, 40, 30, 1, 32, 32},
      depth_inverse_16x16 = {500, 40, 30, 1, 16, 16},
      depth_inverse_32x32 = {500, 40, 30, 1, 32, 32},
      depth_log_16x16     = {500, 40, 30, 1, 16, 16},
      depth_log_32x32     = {500, 40, 30, 1, 32, 32},
      flow_uv_16x16       = {500, 40, 29, 2, 16, 16},
      flow_uv_32x32       = {500, 40, 29, 2, 32, 32},
      flow_ra_16x16       = {500, 40, 29, 2, 16, 16},
      flow_ra_32x32       = {500, 40, 29, 2, 32, 32}
  }
  
  local osname = paths.uname()
  if osname== 'Darwin' then
    sztable = { 
      image_gray_16x16    = {50, 40, 30, 1, 16, 16}, 
      image_gray_32x32    = {50, 40, 30, 1, 32, 32},
      depth_normal_16x16  = {50, 40, 30, 1, 16, 16},
      depth_normal_32x32  = {50, 40, 30, 1, 32, 32},
      depth_inverse_16x16 = {50, 40, 30, 1, 16, 16},
      depth_inverse_32x32 = {50, 40, 30, 1, 32, 32},
      depth_log_16x16     = {50, 40, 30, 1, 16, 16},
      depth_log_32x32     = {50, 40, 30, 1, 32, 32},
      flow_uv_16x16       = {50, 40, 29, 2, 16, 16},
      flow_uv_32x32       = {50, 40, 29, 2, 32, 32},
      flow_ra_16x16       = {50, 40, 29, 2, 16, 16},
      flow_ra_32x32       = {50, 40, 29, 2, 32, 32}
    }
  end
  return sztable[key]
  
end

----------
-- Input: 
-- dtype : image/depth/flow
-- ctype : gray/inverse/normal/log/log/uv/ra
-- res   : 16 | 32
-- Output:
-- data  : a tensor
----------
local function LoadDataset (dtype, ctype, res)
  
  -- filename 
  local filename = DatasetFilename(dtype, ctype, res)
  
  -- size  
  local sz = DatasetSize(dtype, ctype, res)
  
  -- number of elements
  local n = 1
  for i, v in ipairs(sz) do n = n * v  end
  
  -- for debug
  -- print('filename=' .. filename)
  -- print('size    =' .. table.concat(sz, ','))
  -- print('n       =' .. n)
  
  -- load data in
  local file = torch.DiskFile(filename, 'r')
  file:binary()
  local storage = file:readFloat(n)
  file:close()
  
  -- resize the data 
  local data = torch.FloatTensor(storage, 1, torch.LongStorage(sz))
  
  return data
end


local ols = {
  CacheDir  = CacheDir,
  DatasetSize = DatasetSize,
  DatasetFilename = DatasetFilename,
  LoadDataset = LoadDataset
}
 

return ols


