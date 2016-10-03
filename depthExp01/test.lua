
require'build_ds'
require'Examples'

torch.manualSeed(0)
torch.setdefaulttensortype('torch.FloatTensor')

local data     = build_ds()
local examples = Examples(data)

local function minmax(t)
  local min = 1e10
  local max = 1e-10
  for i=1, #t do
    local m = t[i]:min()
    local n = t[i]:max()
    min = min < m and min or m
    max = max > n and max or n
  end
  return min, max
end


local function load_proc(filename)
  local proc = {}
  local f = torch.DiskFile(filename, 'r'):binary()
  
  
  local t=1
  while not f:hasError() do
    proc[t] = f:readObject()
    t = t+1
  end
  f:close()
  
  return utils.concat(1, proc)
end

local function imsave(filename, tensor)
  local img = tensor:minmax()
  image.save(filename, img)
end

local function visualize_process(filename, Y, cb)
  local YY = load_proc(filename)
  local T = #YY
  local min, max = minmax(YY)
  local nbins = 20
  local centers = 
  
  for t=1,T do
    local et = Y[t] - Y
    local abs = torch.abs(et)
    local hist[t] = torch.histc(et, nbins, min, max)
    
    local fn_pred_img, fn_err_img, fn_hist_txt = cb(t)
    imsave(fn_pred_img, Y[t])
    imsave(fn_err_img, abs)
  end

end