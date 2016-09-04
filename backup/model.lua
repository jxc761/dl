require 'nn'


local function build_model(mtype, config)

-- build model
local builder = {
  naive = build_naive_model
}

return builder[mtype](config)

end

local function build_naive_model(config)

  local args, dinput, doutput, nhidden, r = xlua.unpack(
                                              {config}, 'build_naive_config', 'config={dinput, doutput, nhidden=3|5|7, rate=1.2}',                                             
                                              {arg='dinput', type='number', help='dimension of input'},
                                              {arg='doutput', type='number', help='dimension of output'},
                                              {arg='nhidden', type='number', default=3, help='number of hidden layers'},
                                              {arg='rate', type='number', default=1.2, help='the increasing rate of the number of hidden units'})
                                            
  
  local sztable = { [3] = {dinput, dinput*r, dinput*r*r, dinput*r, doutput},
                    [5] = {dinput, dinput*r, dinput*r*r, dinput*r*r*r, dinput*r*r, dinput*r, doutput},
                    [7] = {dinput, dinput*r, dinput*r*r, dinput*r*r*r, dinput*r*r*r*r, dinput*r*r*r, dinput*r*r, dinput*r, doutput}}
  
  local sz = sztable[nhidden]
  local md = nn.Sequential()
  for i = 1, #sz-1 do
    md:add(nn.Linear(sz[i], sz[i+1]))
    md:add(nn.ReLU())
  end
  
  local criterion = nn.MSECriterion()
  
  return md, criterion
end

return build_model