

local function visualize_process(filename, Y, cb)
  local YY = load_proc(filename)
  local T = #YY

  -- for histc 
  local min, max = minmax(YY)
  local nbins = 20
  local bw = (max-min)/bins 
  local centers = torch.linespace(min+0.5*bw, max-0.5*bw, nbins)
  
  for t=1,T do
    local et = Y[t] - Y
    local abs = torch.abs(et)
    local hist = torch.histc(et, nbins, min, max)
    
    local fn_pred_img, fn_err_img, fn_hist_txt = cb(t)
    imsave(fn_pred_img, Y[t])
    imsave(fn_err_img, abs)
    imsave(fn_hist_txt, centers, hist)
  end

end


function loadResult(filename)
  local f=torch.DiskFile(filename, "r"):binary()
  local smp = f:readObject()
  local trc = f:readObject()
  f:close()
  print(smp:size())
  print(trc:size())
end

local inputdir="../../buffer/depthExp01/d01_m01_t01"
local filename=inputdir .. "/examples.dat"
local f=torch.DiskFile(filename, "r"):binary()
smp=f:readObject()
trc=f:readObject()
f:close()

filename=inputdir .. "/result_4.dat"


