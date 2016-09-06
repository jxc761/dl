require 'gnuplot'
require 'paths'


print(arg)

cmd = torch.CmdLine()


local ols = require 'ols'
print(ols.DataDir())

-- local plot_script = [[
--    set term png
--    set output "/Users/Jing/Dropbox/dev/depth/test.png"
--    plot sin(x)
--    set title "test"
--    set grid
--    set output 
--  ]]
--  cmd = 'echo hello world'
-- 
--print(paths.uname())
--
--
--
--gnuplot.setgnuplotexe('/usr/local/bin/gnuplot')
--gnuplot.setterm('x11')
--gnuplot.raw(plot_script)
--/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:/Library/TeX/texbin:/opt/X11/bin
