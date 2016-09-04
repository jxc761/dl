require'torch'
require 'gnuplot'
utils = require 'utils'


function test(arg1, arg2, ...)


end
osname, arch = utils.getOS()

print( osname )
test(1, 2, 3, 4, 5)

require 'Monitor'

local testX = torch.rand(3,2)
local testY = torch.rand(3,2)
local expX  = torch.rand(3,2)
local expY  = torch.rand(3,2)
local prefix = './test'
local m = Monitor(testX, testY, expX, expY, prefix)
print(m)
print(m.prefix)
print(m.fnerr)
print(m.fnexp) 
fn_err, fn_exps = Monitor.filenames('./test')
print(fn_err)
print(fn_exps)