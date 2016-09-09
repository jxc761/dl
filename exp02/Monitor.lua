require 'torch'
require 'paths'
require 'gnuplot'


local Monitor = torch.class('Monitor')

function Monitor:__init(config)

  self.fn_txt = config.fn_txt
  self.fn_img = config.fn_img
  
  self.ftxt = nil
end


function Monitor:start()
  
  self.evals     = {}
  self.durations = {}
  self.epochs    = {} 
  
  if self.fn_txt then
    self.ftxt = assert( io.open(self.fn_txt, 'w'))
    self.ftxt:write(string.format('%16s%16s%16s\r\n', '#epoch', 'time(s)', 'fval') )
  end
  
  if self.fn_img then
   
    -- prepare gnuplot environment
    local os = paths.uname()
    if os == 'Darwin' then
        gnuplot.setgnuplotexe('/usr/local/bin/gnuplot')
        gnuplot.setterm('x11') 
    end
    
    local ext = paths.extname(fn_img)
    local terms = { png='png', svg='svg', eps='postscript eps' }
    local script = string.format("set term %s; set output '%s';", terms[ext], fn_img)
    gnuplot.raw(script)  
  end
  
end





function Monitor:monitor(cur_epoch, cur_duration, cur_loss)
  local cur_idx = #evals+1
  
  self.evals[cur_idx] = cur_loss
  self.durations[cur_idx] = cur_duration
  self.epochs[cur_idx] = cur_epoch

  if self.ftxt then
    self.ftxt:write( string.format('%16d%16.2f%16.4e\r\n', cur_epoch, cur_duration, cur_loss) )
    self.ftxt:flush()
  end

  if fn_img then
    gnuplot.raw(string.format("set output '%s'", fn_img) )
    gnuplot.plot(torch.Tensor(evals))
    gnuplot.raw('set output')
  end
  
end 


function Monitor:stop()
  if ftxt then
    ftxt:close()
  end
end


function Monitor:records()
  return {evals = self.evals, durations=self.durations, epochs=self.epochs}
end







