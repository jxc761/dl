
require'Monitor'
require'evaluate'
require'paths'
require'train'

local utils=require'utils'


function run(model, data, examples, trainOpts, root_output)
  local fn_examples= string.format('%s/examples.dat', root_output)
  examples:save(fn_examples)

  local fn_perform = string.format('%s/perform.txt', root_output)
  local fperform = assert( io.open(fn_perform, 'w'))
  fperform:write(string.format('%s\t%s\t%s\t%s\t%s\r\n', 'opts', 'duration', 'train','valid', 'test'))

  
  for i=1, #trainOpts do

    -- reset model
    model:reset()

    -- setup monitor
    local fn_proc_txt = string.format('%s/process_%d.txt', root_output, i)
    local fn_proc_exp = string.format('%s/process_%d.dat', root_output, i)
    local optMonitor = {
        period   = math.floor(trainOpts[i].nIter/50), 
        fn_proc_txt = fn_proc_txt,
        fn_proc_exp = fn_proc_exp
    }
    local monitor = Monitor(model, data, examples, optMonitor)
    print(monitor)

    -- train model
    local tic=sys.tic()
    train(model, data, monitor, trainOpts[i])
    local duration = sys.toc(tic)


    -- evaluate model
    local mse, predSmpY, predTrcY = evaluate(model, data, examples)

    -- save the mes out
    fperform:write(string.format('%s\t%e\t%e\t%e\t%e\r\n', trainOpts[i].key, duration, mse.train, mse.valid, mse.test))  
    fperform:flush()

    -- save result out 
    local fn_result = string.format('%s/result_%d.dat', root_output, i)
    local fresult   = torch.DiskFile(fn_result, 'w'):binary()
    fresult:writeObject(predSmpY)
    fresult:writeObject(predTrcY)
    fresult:close()

    -- save model out
    local fn_model = string.format('%s/model_%d.dat', root_output, i)
    model:save(fn_model)
  end

  fperform:close()
  
end

