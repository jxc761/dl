require 'torch'
require 'paths'
-- require 'gnuplot'


local Monitor = torch.class('Monitor')

function Monitor:__init(model, data, config)
  self.model = model
  self.data  = data

  self.period  = config.period 
  self.exps    = config.examples
  self.fn_txt  = config.fn_proc_txt
  self.fn_exp  = config.fn_proc_exp

  -- self:_setExpSmp(config.expSmpIdx)


  self.ftxt    = nil
  self.fexp    = nil
  self.epoch    = 0
  self.evals    = nil
  self.durations= nil
end

function Monitor:start()
  self.epoch = 0

  self.evals     = {}
  self.durations = {}

  self.tic       = sys.tic 

  if self.fn_txt then
    self.ftxt = assert( io.open(self.fn_txt, 'w'))
    self.ftxt:write(string.format('%s\t%s\t%s\t%s\r\n', '#epoch', 'period', 'time(s)', 'fval') )
  end

  if self.fn_exp then
    -- self.fdat = assert( io.open(self.fn_dat, 'w') )
    self.fexp = torch.DiskFile(self.fn_exp, 'w'):binary()
  end

end


function Monitor:monitor()

  self.epoch = self.epoch + 1 
  if self.epcho % self.period ~=1 then
    return
  end

  local t=#self.evals + 1 
  self.evals[t]     = model:forward(self.data:validX(), self.data:validY() )
  self.durations[t] = sys.toc(self.tic) -- cur_duration

  if self.ftxt then
    self.ftxt:write( string.format('%16d%16.2f%16.4e\r\n', self.epoch, t, self.durations[t], self.evals[t]) )
    self.ftxt:flush()
  end

  if self.fexp then
    local Y = self.model:predict(self.exps:smpX())
    self.fexp:writeObject(predictY)
  end

end 

function Monitor:stop()
  if self.ftxt then self.ftxt:close() end 
  if self.fexp then  self.fexp:close() end 
end



function Monitor:report()
  return {evals = self.evals, durations=self.durations, epochs=self.epochs}
end






