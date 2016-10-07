

require 'torch'
require 'nn'

require 'cunn'
require 'cutorch'

local Model=torch.class('Model')

---
-- dimIn
-- dimOut
-- nHiddenLayer
-- 
-- type = 'fc' |  'conv'
-- 
function Model:__init(layers)

  self.network = self:build_network(layers)
  self.criterion = nn.MSECriterion()
  

  self.network:cuda()
  self.criterion:cuda()

end

function Model:build_network(layers)

	local model = nn.Sequential()
	for i=1, #layers do 
		local ly = layers[i]
		if ly.type == 'fc' then
			model:add(nn.Linear(ly.dimIn, ly.dimOut)) 
		elseif ly.type == 'conv' then
			local t= torch.Tensor({ly.dimIn, ly.dimOut, ly.kw, ly.kh, ly.dw, ly.dh, ly.pw, ly.ph})
			print(t)
			model:add(nn.SpatialConvolutionMM(ly.dimIn, ly.dimOut, ly.kw, ly.kh, ly.dw, ly.dh, ly.pw, ly.ph))
		elseif ly.type == 'relu' then
			model:add(nn.ReLU())
		else
			error('Unkown network')
		end
	end
	return model 
end

function Model:reset()
	self.network:reset()
	self.param, self.gradParams = self.network:getParameters()
end


function Model:getParameters()
	return self.param, self.gradParams
end


function Model:updateGrad(x, y)
  self.gradParams:zero()
  
  x = x:cuda()
  y = y:cuda()

  local o   = self.network:forward(x)          -- output of the network   
  local l   = self.criterion:forward(o, y)     -- loss of the model
  local dl  = self.criterion:backward(o, y)    -- d_loss/d_output
  
  self.network:backward(x, dl)                 -- d_loss/d_parameters and d_loss / d_x
  return l, self.gradParams
end



function Model:forward(X, Y)
  X = X:cuda()
  Y = Y:cuda()

  local step= 1000
  local sum = 0

  local n = X:size(1)
  for index = 1, n, step do
    local size = step < (n-index+1) and step or (n-index+1) 
    local x = X:narrow(1, index, size)
    local y = Y:narrow(1, index, size)
    local l = self.criterion:forward(self.network:forward(x), y)
    sum = sum + l * size
  end
  
  local result = sum / n;
  return result
end


function Model:predict(x)
    x = x:cuda()
	return self.network:forward(x):float()
end


function Model:__tostring()
  local t = {}
  t[1] = self.network:__tostring()
  t[2] = self.criterion:__tostring()
  return table.concat(t, '\r\n')
end

function Model:save(filename)
	torch.save(filename, self.network:float())
end

--[[


function Model:build_fc_network(dimIn, dimOut, nHiddenLayer)
	local model = nn.Sequential() 
	-- first layer 
	model:add( nn.Linear(dimIn, dimIn * 1.5) )
	model:add( nn.ReLU() )
	for i = 1, nHiddenLayer+1 do
		model:add(sz[i], sz[i+1])
		model:add(nn.ReLU())
	end
	return model
end


function Model:build_conv_network(dimIn, dimOut, nHiddenLayer)

	local model = nn.Sequential()
	local dims = torch.LongTensor(2+nHiddenLayer, 1):fill(256) 
	dims[1] = dimIn
	dims[2+nHiddenLayer] = dimOut 
	for i=1, nHiddenLayer+1 do
		-- SpatialConvolutionMM(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
		model:add( nn.SpatialConvolutionMM(dims[i], dims[i+1]), 5, 5, 1, 1, 2, 2)
		model:add( nn.ReLU() )
	end
end

]]


