require 'OLSDataset'


---
--
--OLSDataSource(config)
--OLSDataSource(input, output, config)
--OLSDataSource(input_dtype, input_ctype, input_res, 
--             output_dtype, output_ctype, output_res,  
--             nTrainScenes, nValidScenes, nTestScenes, 
--             trainExpIdx, validExpIdx, testExpIdx,
--             [, batchsz])
---
local function OLSDataSource(...)

local input, output, config = parseInput({...})

local trainX, validX, testX = buildOLSDataset(input, config)
local trainY, validY, testY = buildOLSDataset(output, config) 

local trainExpX  = trainX:index(config.trainExpIdx)
local trainExpY  = trainY:index(config.trainExpIdx)
local validExpX  = validX:index(config.validExpIdx)
local validExpY  = validY:index(config.validExpIdx)
local testExpX   = testX:index(config.testExpIdx)
local testExpY   = testY:index(config.testExpIdx)

return { train={ X=trainX, Y=trainY, expX = trainExpX, expY = trainExpY},
         valid={ X=validX, Y=validY, expX = validExpX, expY = validExpY},
         test ={ X=testX,  Y=testY,  expX = testExpX,  expY = testExpY } }
          
--return { trainX = trainX, trainY = trainY, trainExpX = trainExpX, trainExpY = trainExpY, 
--         validX = validX, validY = validY, validExpX = validExpX, validExpY = validExpY, 
--         testX = testX, testY = testY, testExpX = testExpX, testExpY = testExpY}
end

local function buildOLSDataset(dconf, sconf)
  local dtype = data_config.dtype
  local ctype = data_config.ctype
  local offset1 = 1
  local offset2 = sconf.nTrainScenes + 1
  local offset3 = sconf.nValidScenes + sconf.nTrainScenes + 1
  
  local m = sconf.n
  local data = ols.LoadDataSet(dconf.dtype, dconf.ctype, dconf.res)
  local trainset = OLSDataset( { org_data=data, dtype=dconf.dtype, ctype=dconf.ctype,
                                 startscene=offset1, nscenes=sconf.nTrainScenes, batchsz=sconf.batchsz})

  local validset = OLSDataset( { org_data=data, dtype=dconf.dtype, ctype=dconf.ctype,
                                 startscene=offset2, nscenes=sconf.nValidScenes, batchsz=sconf.batchsz})
  
  local testset =  OLSDataset( { org_data=data, dtype=dconf.dtype, ctype=dconf.ctype,
                                 startscene=offset3, nscenes=sconf.nTestScenes, batchsz=sconf.batchsz}) 
                                                
  return trainset, validset, testset                               
end

local function parseInput(...)


end