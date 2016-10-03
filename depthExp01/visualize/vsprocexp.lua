require 'paths'
require 'image'
require '../utils'


function load_process(filename)
  local result={}
  local f = torch.DiskFile(filename, "r"):binary()
  f:quiet()
  while not f:hasError() do
    result[#result+1] =  f:readObject()
  end
  f:close()
  return utils.concat(0, result)
end


local root   = '/Users/Jing/Dropbox/dev/dl/buffer/depthExp01/d01_m01_t02'
local fn_proc= string.format('%s/process_1.dat', root)
local result = load_process(fn_process)

local T=result:size(1)
local N=result:size(2) 
local min=result:min()
local max=result:max()
for i=1,N do
  local output = string.format('%s/process1_smp%d', root_output, i)
  paths.mkdir(output)
  local Y = result:narrow(2, i, 1)
  local min = Y:min()
  local max = Y:max()
  for t=1,T do
    local im = image.minmax({tensor=result[t], min=min, max=max})
    local fn = string.format('%s/predict_%d.png', output, t)
    image.save(fn, im)
  end

end
