require 'torch'
require 'image'
require 'paths'

require 'OLSDataset'

local ols = require 'ols'
local utils = require 'utils'



local function testOLSDataset()

	local data = ols.LoadDataset('image', 'gray', 32)
	local subset = data:narrow(1, 1, 30)
	local output = './tmp'
  paths.mkdir(output)
  

  local ds = OLSDataset({data = subset, step=2, stride=3, speed=2, bidirect=true})
  -- local ds = OLSDataset({data = subset, step=1, stride=1, speed=1, bidirect=false})
	print(ds)

--	local idx = {1, 2, 13, 14, 13*40 + 1, 13*40 + 2, 13*40 + 13, 13*80 + 1}
--	local samples = ds:index(idx)
--	local images = samples:view(#idx, ds.stride, ds.C, ds.W, ds.H):permute(1, 2, 3, 5, 4)
--
--	for i = 1,#idx do
--		for j = 1,ds.stride do
--			local filename = string.format('%s/%d_%d.png', output, idx[i], j) 
--			image.save(filename, images[i][j])
--		end
--	end
	
	

	idx = {1, 41, 81}
	
	local traces = ds:traces(idx)
	
  for i=1, #idx do
	  local trace = traces[i] 
	  print(trace:size())   
	  for j = 1, ds.M do
	    local sample = trace[j]:view(ds.stride, -1)
	    for k= 1, ds.stride do
	     
	      local filename = string.format('%s/%d_%d_%d.png', output, idx[i], j, k)
	      image.save(filename, sample[k]:view(ds.C, ds.W, ds.H):permute(1,3,2))
	    end
	  end
	end

end 


testOLSDataset()