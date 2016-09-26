require 'torch'
require 'paths'
require 'gnuplot'


local Monitor = torch.class('Monitor')

function Monitor:__init( model, data, config)
  self.model = model
  self.data  = data


  self.fn_txt  = config.fn_process_txt
  self.fn_dat  = config.fn_process_dat

  self.ftxt    = nil
  self.fdat    = nil

  self.epoch    = nil
  self.evals    = nil
  self.durations= nil
  

end


function Monitor:start()

  self.epoch     = 0
  self.evals     = {}
  self.durations = {}

  self.tic       = sys.tic 

  if self.fn_txt then
    self.ftxt = assert( io.open(self.fn_txt, 'w'))
    self.ftxt:write(string.format('%16s%16s%16s\r\n', '#epoch', 'time(s)', 'fval') )
  end

  if self.fn_dat then
    self.fdat = assert( io.open(self.fn_dat, 'w') )
  end


end

function Monitor:monitor()

  self.epoch = self.epoch + 1 
  self.evals[self.epoch]     = model:forward(self.data:validX(), self.data:validY() )
  self.durations[self.epoch] = sys.toc(self.tic) -- cur_duration

  if self.ftxt then
    self.ftxt:write( string.format('%16d%16.2f%16.4e\r\n', self.epoch, self.durations[self.epoch], self.evals[self.epoch]) )
    self.ftxt:flush()
  end

  if self.fdat then
    self.fwrite()
  end

  
end 


function Monitor:stop()
  if ftxt then
    ftxt:close()
  end
end



function Monitor:report()
  return {evals = self.evals, durations=self.durations, epochs=self.epochs}
end






