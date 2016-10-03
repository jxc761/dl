require 'paths'
require 'os'
require 'math'
require 'torch'


utils = {}

function utils.project_dir()
  local osname = paths.uname()
  
  local path = {
    Darwin = '/Users/Jing/Dropbox/dev/dl',
    Linux  = '/home/jxc761/dl'
  }
   
  return path[osname]
  
end

function utils.buffer_dir()
  local osname = paths.uname()
  
  local path = {
    Darwin = '/Users/Jing/Dropbox/dev/dl/buffer',
    Linux  = '/home/jxc761/dl/buffer'
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
  return result
end

function utils.unpack(t, fields)
  print(t)
  print(fields)
  local result = {}
  for i = 1, #fields do
    result[i] = t[fields[i]]
  end
  
  return table.unpack(result)

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



function utils.catv(...)
  local v={}
  local a={...}
  local ii=1
  for i=1, #a do
    for j=1, utils.nElement(a[i]) do
      v[ii] = a[i][j]
      ii=ii+1
    end
  end
  return v
end




function utils.concat(dim, ...)
  local a = {...}
  if (type(a[1]) == "table")  then
    a = { table.unpack(a[1]) }
  end

  -- expand one dim
  if dim==0 then
    for i=1, #a do
      local oldsz = a[i]:size()
      local newsz = utils.catv({1}, oldsz)
      a[i] = a[i]:view(torch.LongStorage(newsz))
    end
    dim = 1
  end
  
  -- remove empty tensor
  for i=#a,1,-1 do
    if a[i]:dim() < dim then
      table.remove(a, i)
    end
  end
  
  -- initialize 
  local result = #a and a[1]
  for i=2,#a do
    result=torch.cat(result, a[i], dim)  
  end
  
  return result 
end



return utils


