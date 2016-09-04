ols = require 'ols'


local function DataSource(inputkey, outputkey, configkey)

  configkey = configkey or 'debug_config'
  
  local datasets  = {
     image_gray_16     = { dtype = 'image', ctype ='gray', res=16},
     image_gray_32     = { dtype = 'image', ctype ='gray', res=32},
     depth_inverse_16  = { dtype = 'depth', ctype='inverse', res=16},
     depth_inverse_32  = { dtype = 'depth', ctype='inverse', res=32},
  }
  
  local debug_config = { nTrainScenes=40, nValidScenes=5, nTestScenes=5, 
                           trainExpIdx={1, 10, 100}, validExpIdx={1, 10, 100}, testExpIdx={1, 10, 100} }
                           
  local release_config = { nTrainScenes=400, nValidScenes=50, nTestScenes=50, 
                           trainExpIdx={1, 10, 100, 1000}, validExpIdx={1, 10, 100, 1000}, testExpIdx={1, 10, 100, 1000} }
  local confs = { debug = debug_config, release = release_config }
                 
  return  OLSDataSource(datasets[inputkey], dataset[outputkey], confs[configkey])
  
end



