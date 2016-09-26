require 'paths'
require 'os'
require 'math'
require 'torch'
local utils = {}


function utils.project_dir()
  local osname = paths.uname()
  
  local path = {
    Darwin = '/Users/Jing/Dropbox/dev/dl',
    Linux  = '/home/jxc761/dl'
  }
   
  return path[osname]
  
end


---
-- merge t2 to t1
-- t2 will override the same field of t1
function utils.merge(t1, t2)

  for key, value in pairs(t2) do 
    t1[key] = value
  end
  return t1
end

function utils.extend(target, ...)
  local tables = {...}
  
  local n = tables and #tables or 0
  for i=1, n do
    utils.merge(target, tables[i])
  end
  
  return target
  
end

function utils.getfields(t, fields)
  local result = {}
  for i = 1, #fields do
    result[fields[i]] = t[fields[i]]
  end
end


function utils.checkargs(options, defaults, required)

  local opts = utils.extend({}, defaults, options)
  
  -- check if all required arguments exists
  local n = required and #required or 0
  for i=1, n do
    assert( opts[required[i]], "Miss required argument: " .. required[i])
  end
  
  return opts
  
end

function utils.nElement(input) 
  if torch.isTensor(input) then
    return input:nElement()
  else
    return #input
  end
    
end


function utils.sub2ind(sub, sz)
  local n = #sz
  local ind = sub[1] - 1

  for i = 2, n do
    ind = ind * sz[i] + sub[i] - 1
  end

  return ind + 1
end


function utils.ind2sub(ind, sz)
  local sub = {}
  local d = ind - 1
  local n = #sz

  for i = n, 1, -1 do
    sub[i] = d % sz[i] + 1
    d = math.floor(d/sz[i])
  end

  return table.unpack(sub)
end




return utils


