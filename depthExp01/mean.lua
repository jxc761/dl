require 'torch'
require 'image'
require 'paths'

local ols = require 'ols'

local function mean(dtype, ctype, res, bOnlyValid)

  local X = ols.LoadDataset(dtype, ctype, res)


  if bOnlyValid then
    local idx = ols.ValidSceneIdx()
    X = X:index(1, idx)
  end

  local sz = X:size()
  local n = sz[1] * sz[2] * sz[3]
  local d = sz[4] * sz[5] * sz[6]
  print('n=' .. n ..', d=' .. d)

  local mu = X:view(n, d):mean(1) : view(sz[4], sz[5], sz[6])


	
  return mu
end


local function process(dtype, ctype, res)

	local mu = mean(dtype, ctype, res, true)

	local output = '../buffer/mean'
	paths.mkdir(output)


	local fn_img = string.format('%s/%s_%s_%dx%d_mean.png', output, dtype, ctype, res, res)
	local fn_dat = string.format('%s/%s_%s_%dx%d_mean.dat', output, dtype, ctype, res, res)


	local fdat = torch.DiskFile(fn_dat, 'w'):binary()
	fdat:writeObject(mu)
	fdat:close()

	local img = image.minmax({tensor=mu})
	image.save(fn_img, img)

end

process('depth', 'inverse', 16) 
process('depth', 'inverse', 32) 

process('image', 'gray', 16)
process('image', 'gray', 32)

process('depth', 'normal', 16) 
process('depth', 'normal', 32) 
process('depth', 'log', 16) 
process('depth', 'log', 32) 

