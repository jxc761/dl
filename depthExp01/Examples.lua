local utils = require'utils'

local Examples=torch.class('Examples')



local function getSamplesFromDataset(data, dataset, idx)
	local X = data:indexX(idx, dataset)
	local Y = data:indexY(idx, dataset)
	local keys = {}
	for i=1, #idx do
		keys[i] = string.format('smp_%s_%d', dataset, idx[i])
	end
	return X, Y, keys
end



local function getTracesFromDataset(data, dataset, idx)
	local X = data:TracesX(idx, dataset)
	local Y = data:TracesY(idx, dataset)
	local xtrace = data:traces(idx, dataset)
	local ytrace = data:traces(idx, dataset)

	local keys = {} 
	for i=1, #idx do
		keys[i] = string.format('trc_%s_%d', dataset, idx[i])
	end

	return X, Y, xtrace, ytrace, keys
end

local function getExpSamples(data, idx)
	-- local idx = torch.LongStorage({1, 500, 1000, 1500})
	local X1, Y1, keys1 = getSamplesFromDataset(data, 'train', idx)
	local X2, Y2, keys2 = getSamplesFromDataset(data, 'valid', idx)
	local X3, Y3, keys3 = getSamplesFromDataset(data, 'test', idx)

	local X = utils.concat(1, X1, X2, X3)
	local Y = utils.concat(1, Y1, Y2, Y3)
	local keys = utils.catv(keys1, keys2, keys3)
	-- return X, Y, keys
	return {X=X, Y=Y, keys=keys}
end


local function getExpTraces(data, idx)

	-- local idx = torch.LongStorage({1, 40, 400})
	local trainX   ={}
	local trainY   ={}
	local trainKey ={}
	local X1, Y1, xtrace1, ytrace1, keys1 = getTracesFromDataset(data, 'train', idx)
	local X2, Y2, xtrace2, ytrace2, keys2 = getTracesFromDataset(data, 'valid', idx)
	local X3, Y3, xtrace3, ytrace3, keys3 = getTracesFromDataset(data, 'test', idx)

	local X = utils.concat(1, X1, X2, X3)
	local Y = utils.concat(1, Y1, Y2, Y3)
	local xtraces = utils.concat(1, xtrace1, xtrace2, xtrace3)
	local ytraces = utils.concat(1, ytrace1, ytrace2, ytrace3)
	local keys = utils.catv(keys1, keys2, keys3)

	return {X=X, Y=Y, xtraces=xtraces, ytraces=ytraces, keys=keys}
	-- return X, Y, xtraces, ytraces, keys

end



function Examples:__init(data)
	
	local smpIdx = torch.LongStorage({1, 500, 1000, 1500})
	local trcIdx = torch.LongStorage({1, 40, 400})
	self.samples = getExpSamples(data, smpIdx)  -- X, Y, keys
	self.traces  = getExpTraces(data, trcIdx)   -- X, Y, xtraces, ytraces, keys
	self.nSample = self.samples.X:size(1)
	self.nTrace  = self.traces.X:size(1)
end


function Examples:smpX()
	return self.samples.X
end

function Examples:smpY()
	return self.samples.Y
end

function Examples:smpKeys()
	return self.samples.keys
end

function Examples:trcX()
	return self.traces.X
end

function Examples:trcY()
	return self.traces.Y
end

function Examples:trcKeys()
  return self.traces.keys
end

function Examples:save(filename)
  local fexp = torch.DiskFile(filename, 'w'):binary()
  fexp:writeObject(self.examples)
  fexp:writeObject(self.traces)
  fexp:close()
end

function Examples:__tostring()
  local t={}
  t[1] = string.format('nExpSmp=%d', self.samples.X:size(1))
  t[#t+1] = string.format('nExpTrc=%d', self.traces.X:size(1))
  t[#t+1] = string.format('samples: ')
  t[#t+1] = table.concat(self.samples.keys, '\r\n')
  t[#t+1] = 'traces:'
  t[#t+1] = table.concat(self.traces.keys, '\r\n')
  return table.concat(t, '\r\n')

end
