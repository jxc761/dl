require 'torch'

local Monitor = torch.class('Monitor')

function Monitor:__init(testX, testY, expX, expY, prefix)

  self.testX = testX
  self.testY = testY
  self.expX  = expX
  self.expY  = expY
  self.prefix = prefix
  

  self.fnerr = string.format('%serrs.txt', prefix)
  self.fnexp = string.format('%sexps.txt', prefix)
  
  self.fnexpX = string.format('%sexpX.dat', prefix)
  self.fnexpY = string.format('%sexpY.dat', prefix)
  self.fnests = string.format('%sests.dat', prefix)
  
  
end


function Monitor:start()
  self.ferr = assert( io.open(self.fnerr, 'w'))
  self.fexp = assert( io.open(self.fnexp, 'w'))
  self._it = 0
  self.estYs = {}
  
  torch.save(self.fnexpX, self.expX, 'binary')
  torch.save(self.fnexpY, self.expY, 'binary')
end

function Monitor:monitor(iter, md, criterion)
  self._it = self._it+1

  -- evaluate on the testing data set
  local o = md:forward(self.testX)
  local l = criterion:forward(o, self.testY)
  self.ferr:write( string.format('%16d%16d\t%16.2e\r\n', self._it, iter, l) )
  
  -- evaluate on the examples
  local n = size(self.expX, 1) 
  local estY = md:foward(self.expX)

  
  self.exp:write( string.format('%16d%16d', self._it, iter) )
  for i = 1, size(estY, 1) do
    local li = criterion:forward(estY[{i}], self.expY[{i}])
    self.fexp:write( string.format('\t%16d', li) )
  end
  self.fexp:write('\r\n')
  
  self.estYs[#estYs+1] = estY
  torch.save(self.fnests, self.estYs, 'binary')
end


function Monitor:stop()
  self.ferr:close()
  self.fexp:close()
  self._it = 0
end






