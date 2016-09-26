
-- 12 settings
local function conf_data(i, j)

  local settings = {
    {res=16, stride=2, speed=2, step=1}, 
    {res=16, stride=3, speed=1, step=1}, 
    {res=32, stride=2, speed=2, step=1}, 
    {res=32, stride=3, speed=1, step=1}
  }

  local target_ctypes = { 'log', 'inverse', 'normal'}
  
  local mean = function(x) 
      return torch.mean(x, 1)
  end

  return {
    input ={ dtype='image', ctype='gray', res=res, step=step, stride=stride, speed=speed, preprocess='nil'},
    target={ dtype='depth', ctype=target_ctype, res=res, step=step, stride=stride, speed=speed, preprocess={func=mean, dimOut=res*res}},
    nTrain=395,
    nValid=20,
    nTest =20,
    bExclude=false
  }
  
end



-- 4 different settings
local function conf_model(i, f, d, l, dimx, dimt)

  local conv_layers1 = {
    {type='conv', dimIn=f, dimOut=d, kw=9, kh=9, dw=1, dh=1, pw=4, ph=4}, {type='relu'},
    {type='conv', dimIn=d, dimOut=d, kw=9, kh=9, dw=1, dh=1, pw=4, ph=4}, {type='relu'},
    {type='conv', dimIn=d, dimOut=d, kw=7, kh=7, dw=1, dh=1, pw=3, ph=3}, {type='relu'},
    {type='conv', dimIn=d, dimOut=d, kw=7, kh=7, dw=1, dh=1, pw=3, ph=3}, {type='relu'},
    {type='conv', dimIn=d, dimOut=d, kw=5, kh=5, dw=1, dh=1, pw=2, ph=2}, {type='relu'},
    {type='conv', dimIn=d, dimOut=l, kw=5, kh=5, dw=1, dh=1, pw=2, ph=2}, {type='relu'},
  }

  local conv_layers2 = {
    {type='conv', dimIn=f, dimOut=d, kw=7, kh=7, dw=1, dh=1, pw=3, ph=3}, {type='relu'},
    {type='conv', dimIn=d, dimOut=d, kw=7, kh=7, dw=1, dh=1, pw=3, ph=3}, {type='relu'},
    {type='conv', dimIn=d, dimOut=d, kw=5, kh=5, dw=1, dh=1, pw=2, ph=2}, {type='relu'},
    {type='conv', dimIn=d, dimOut=d, kw=5, kh=5, dw=1, dh=1, pw=2, ph=2}, {type='relu'},
    {type='conv', dimIn=d, dimOut=d, kw=3, kh=3, dw=1, dh=1, pw=1, ph=1}, {type='relu'},
    {type='conv', dimIn=d, dimOut=l, kw=3, kh=3, dw=1, dh=1, pw=1, ph=1}, {type='relu'},
  }

  local fc_layers1 = {
    {type='fc', dimIn=dimx,     dimOut=math.floor(1.5*dimx)}, {type='relu'},
    {type='fc', dimIn=math.floor(1.5*dimx), dimOut=math.floor(3.0*dimx)}, {type='relu'},
    {type='fc', dimIn=math.floor(3.0*dimx), dimOut=math.floor(3.0*dimt)}, {type='relu'},
    {type='fc', dimIn=math.floor(3.0*dimt), dimOut=math.floor(1.5*dimt)}, {type='relu'},
    {type='fc', dimIn=math.floor(1.5*dimt), dimOut=math.floor(dimt)}, {type='relu'},
  }

  local fc_layers2 = {
    {type='fc', dimIn=dimx,     dimOut=math.floor(1.5*dimx)}, {type='relu'},
    {type='fc', dimIn=math.floor(1.5*dimx), dimOut=math.floor(3.0*dimx)}, {type='relu'},
    {type='fc', dimIn=math.floor(3.0*dimx), dimOut=math.floor(3.0*dimx)}, {type='relu'},
    {type='fc', dimIn=math.floor(3.0*dimx), dimOut=math.floor(3.0*dimx)}, {type='relu'},
    {type='fc', dimIn=math.floor(3.0*dimx), dimOut=math.floor(dimt)}, {type='relu'},
  }

  local layers = {conv_layers1, conv_layers2, fc_layers1, fc_layers2}

  return layers[i]
end


-- index = 12 * 4 (48)
function conf(index) 
  local opts = {}
  local sub = utils.ind2sub(index, {4, 3, 4})

  opts.data = conf_data(sub[1], sub[2])
  
  local f = opts.data.input.stride 
  local l = 1
  local d =256
  local dimx = opts.data.input.res *  opts.data.input.res * opts.data.input.stride
  local dimt = opts.data.target.res * opts.data.target.res
  opts.eval = { 
    train_expsmp_idx = {1, 100, 1000}, 
    valid_expsmp_idx = {1, 100, 1000}, 
    test_expsmp_idx  = {1, 100, 1000},
    train_exptrc_idx = {1, 100, 1000},
    test_exptrc_idx  = {1, 100, 1000}
  }
  
  opts.model = conf_model(sub[3], f, d, l, dimx, dimy)


  opts.learningrates = {1, 1e-1, 1e-2, 1e-3, 1e-4}
  opts.output = string.format('%s/buffer/depthExp01/step1/exp%02d', utls.project_dir(), index)
  paths.mkdir(opts.output)

  return opts
end